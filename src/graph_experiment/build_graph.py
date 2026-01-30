
import json
import re
from pathlib import Path
import glob

def parse_and_build_graph(batch_dir: Path):
    """
    Parses all 'batch_output_*.jsonl' files in the given directory,
    reconstructs the graph data for each document, and saves it.
    """
    print(f"\n--- Building Graph from Batch Results in {batch_dir} ---")
    
    # Structure: { "doc_name_directory": { "entities": [], "relationships": [] } }
    docs_graph_data = {}
    
    # 1. scan for output files
    # output_files = list(batch_dir.glob("batch_output_*.jsonl"))
    # glob might not return absolute paths if using pattern relative to cwd?
    # using Path.glob is safer.
    output_files = list(batch_dir.glob("batch_output_*.jsonl"))
    
    if not output_files:
        print("No batch output files found.")
        return
        
    print(f"Found {len(output_files)} output files to process.")
    
    # Helper to find matching directory for a document name
    processed_dir = batch_dir.parent # data/processed_experiment
    # Pre-scan actual directories to create a map: safe_name -> Path
    # Because custom_id uses safe_name (spaces replaced by underscores)
    dir_map = {}
    for d in processed_dir.iterdir():
        if d.is_dir() and d.name != "batch_input":
            # Same logic used in prepare_batch:
            # safe_doc_name = doc_dir.name.replace(" ", "_").replace("[", "").replace("]", "")
            safe_name = d.name.replace(" ", "_").replace("[", "").replace("]", "")
            dir_map[safe_name] = d
            
    print(f"Mapped {len(dir_map)} target document directories.")

    total_chunks_processed = 0

    for file_path in output_files:
        print(f"Reading: {file_path.name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                    
                try:
                    response = json.loads(line)
                    
                    # 1. Identify Document
                    custom_id = response.get("custom_id", "")
                    if not custom_id: 
                        continue
                        
                    # custom_id format: DOCNAME_INDEX (e.g. Samsung_Report_0)
                    # We need to find the split point.
                    # Heuristic: The last underscore separates index.
                    parts = custom_id.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                        
                    doc_safe_name = parts[0]
                    
                    # Check if this doc exists in our map
                    if doc_safe_name not in dir_map:
                        # try fuzzy match or skip
                        # print(f"Warning: Unknown doc name in ID: {doc_safe_name}")
                        continue
                        
                    target_dir = dir_map[doc_safe_name]
                    target_dir_name = target_dir.name # Key for docs_graph_data
                    
                    # 2. Extract Content
                    if "response" not in response or not response["response"]:
                        continue
                        
                    body = response["response"]["body"]
                    if not body:
                        continue
                        
                    # OpenAI Batch outputs are ChatCompletion objects
                    content_str = body["choices"][0]["message"]["content"]
                    
                    # 3. Clean Markdown
                    # Entities might be wrapped in ```json ... ```
                    if content_str.strip().startswith("```"):
                        # Remove first line (```json) and last line (```)
                        # Or regex replace
                        content_str = re.sub(r"^```[a-zA-Z]*\n", "", content_str.strip())
                        content_str = re.sub(r"\n```$", "", content_str.strip())
                    
                    # 4. Parse JSON
                    extract_result = json.loads(content_str)
                    
                    # 5. Aggregate
                    if target_dir_name not in docs_graph_data:
                        docs_graph_data[target_dir_name] = {"entities": [], "relationships": []}
                    
                    if "entities" in extract_result:
                        docs_graph_data[target_dir_name]["entities"].extend(extract_result["entities"])
                    if "relationships" in extract_result:
                        docs_graph_data[target_dir_name]["relationships"].extend(extract_result["relationships"])
                        
                    total_chunks_processed += 1
                    
                except json.JSONDecodeError:
                    print(f"  ! JSON Decode Error in {custom_id} (Line {line_idx})")
                except Exception as e:
                    print(f"  ! Error processing line {line_idx}: {e}")

    print(f"\nTotal Chunks Processed: {total_chunks_processed}")

    # 6. Save Graph Data
    for doc_dir_name, graph in docs_graph_data.items():
        doc_dir = processed_dir / doc_dir_name
        output_path = doc_dir / "graph_data.json"
        
        # Deduplication could happen here, but for now we dump all extracted items
        # Users might want to deduplicate entities by name later.
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {output_path.name}")
        print(f"  - Document: {doc_dir_name}")
        print(f"  - Entities: {len(graph['entities'])}")
        print(f"  - Relationships: {len(graph['relationships'])}")

if __name__ == "__main__":
    # Assuming standard path
    BATCH_DIR = Path("./data/processed_experiment/batch_input")
    parse_and_build_graph(BATCH_DIR)
