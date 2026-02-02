
import json
import os
from pathlib import Path
from openai import OpenAI
import src.config as config
def process_batch_results(batch_dir: Path):
    client = OpenAI()
    
    # Load submitted jobs record
    record_file = batch_dir / "submitted_jobs.json"
    if not record_file.exists():
        print("No submitted jobs record found.")
        return
        
    with open(record_file, "r") as f:
        jobs = json.load(f)
        
    print(f"Checking status for {len(jobs)} jobs...")
    
    completed_results = []
    
    for job_info in jobs:
        job_id = job_info["job_id"]
        
        # Check status
        batch_job = client.batches.retrieve(job_id)
        print(f"Job {job_id} Status: {batch_job.status}")
        
        if batch_job.status == "completed" and batch_job.output_file_id:
            # Download Results
            print(f"  - Downloading results ({batch_job.output_file_id})...")
            
            result_content = client.files.content(batch_job.output_file_id).content
            
            # Save Raw Output
            output_filename = f"batch_output_{job_id}.jsonl"
            output_path = batch_dir / output_filename
            
            with open(output_path, "wb") as f:
                f.write(result_content)
                
            print(f"  - Saved raw output to: {output_path}")
            
            # Parse and save structured data (Placeholder for now)
            # You would iterate line by line, match custom_id to doc/chunk, and save entities.
            
        elif batch_job.status == "failed" or batch_job.status == "expired" or batch_job.status == "cancelled":
            print(f"  - Job Failed/Expired: {batch_job.errors}")
            retry_failed_job(client, job_info, batch_dir, jobs)

    # Save updated job records if any retries added
    record_file = batch_dir / "submitted_jobs.json"
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    # After processing all jobs, parse and save graph data
    parse_and_save_graph(batch_dir)

def parse_and_save_graph(batch_dir: Path):
    """
    Parses all batch_output_*.jsonl files and aggregates graph data per document.
    """
    print("\n--- Parsing Batch Results & Building Graph ---")
    
    # Dictionary to hold graph data per document
    # Structure: { "doc_name": { "entities": [], "relationships": [] } }
    docs_graph_data = {}
    
    # Find all output files
    output_files = list(batch_dir.glob("batch_output_*.jsonl"))
    
    if not output_files:
        print("No batch output files found to parse.")
        return
        
    for file_path in output_files:
        print(f"Parsing: {file_path.name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    response = json.loads(line)
                    
                    # 1. Extract Custom ID
                    custom_id = response.get("custom_id", "")
                    if not custom_id:
                        continue
                        
                    # Format: DOCNAME_CHUNKINDEX (e.g. 삼성전자_사업보고서_0)
                    # We need to reconstruct the Doc Name. 
                    # Simpler is to use the directories we know exist to match?
                    # Or just split by last underscrore?
                    # Ideally we should have stored a mapping, but let's infer based on existing dirs.
                    
                    # Heuristic: The last part is the index. Join the rest.
                    parts = custom_id.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    doc_name_safe = parts[0]
                    
                    # 2. Extract Content
                    # Response -> body -> choices -> message -> content
                    # Note: Batch API response structure matches Chat Completion response
                    
                    if "response" not in response or not response["response"]:
                        # Failed request within batch?
                        continue
                        
                    response_body = response["response"]["body"]
                    if not response_body:
                        continue
                        
                    content_str = response_body["choices"][0]["message"]["content"]
                    
                    # Clean Markdown Code Blocks (```json ... ```)
                    if content_str.strip().startswith("```"):
                        content_str = content_str.strip().strip("`").replace("json", "", 1).strip()
                    
                    # 3. Parse JSON Content (GraphExtraction)
                    # The LLM output should be a JSON string corresponding to GraphExtraction schema
                    try:
                        graph_data = json.loads(content_str)
                    except json.JSONDecodeError:
                        print(f"    ! Error decoding JSON content for {custom_id}")
                        # Debug: Save failed content to inspect
                        # with open(f"debug_failed_{custom_id}.txt", "w", encoding="utf-8") as df:
                        #     df.write(content_str)
                        continue
                        
                    # 4. Integrate into Document Graph
                    if doc_name_safe not in docs_graph_data:
                        docs_graph_data[doc_name_safe] = {"entities": [], "relationships": []}
                    
                    if "entities" in graph_data:
                        docs_graph_data[doc_name_safe]["entities"].extend(graph_data["entities"])
                    if "relationships" in graph_data:
                        docs_graph_data[doc_name_safe]["relationships"].extend(graph_data["relationships"])
                        
                except Exception as e:

                    print(f"    ! Error parsing line: {e} {content_str}")

    # Save to files
    # We need to map 'doc_name_safe' back to the actual directory name.
    # The 'doc_name_safe' was created by: doc_dir.name.replace(" ", "_").replace("[", "").replace("]", "")
    # So we scan the processed directory to find the matching folder.
    
    processed_dir = batch_dir.parent # data/processed_experiment
    
    for safe_name, graph_content in docs_graph_data.items():
        # Find matching directory
        target_dir = None
        for d in processed_dir.iterdir():
            if not d.is_dir(): continue
            
            candidate_safe = d.name.replace(" ", "_").replace("[", "").replace("]", "")
            if candidate_safe == safe_name:
                target_dir = d
                break
        
        if target_dir:
            output_path = target_dir / "graph_data.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph_content, f, ensure_ascii=False, indent=2)
            print(f"Saved Graph Data: {output_path} (Entities: {len(graph_content['entities'])}, Relationships: {len(graph_content['relationships'])})")
        else:
            print(f"Warning: Could not find original directory for {safe_name}")

def retry_failed_job(client, job_info, batch_dir, jobs_list):
    """
    Splits the failed job's input file into two and resubmits.
    """
    original_file_id = job_info["file_id"]
    original_filename = job_info["file"]
    
    print(f"  > Initiating Retry for {original_filename} (Split & Retry)...")
    
    # 1. Download original input file content
    try:
        content = client.files.content(original_file_id).content.decode("utf-8")
    except Exception as e:
        print(f"    ! Error downloading input file: {e}")
        return

    lines = [line for line in content.splitlines() if line.strip()]
    total_lines = len(lines)
    
    if total_lines < 2:
        print("    ! Cannot split file with less than 2 requests. Skipping.")
        return
        
    mid_point = total_lines // 2
    part_a = lines[:mid_point]
    part_b = lines[mid_point:]
    
    print(f"    > Splitting {total_lines} requests into {len(part_a)} and {len(part_b)}")
    
    # 2. Create new files
    # Determine new filenames (e.g., batch_input_part_1_retry_1.jsonl)
    base_name = os.path.splitext(original_filename)[0]
    
    parts = [("partA", part_a), ("partB", part_b)]
    
    for suffix, data_lines in parts:
        new_filename = f"{base_name}_retry_{suffix}.jsonl"
        new_file_path = batch_dir / new_filename
        
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data_lines))
            
        print(f"    > Created {new_filename}")
        
        # 3. Submit new batch
        with open(new_file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
            
        new_batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Retry of {original_filename}", "original_file": new_filename}
        )
        
        print(f"    > Submitted Retry Job {new_batch_job.id}")
        
        # Add to jobs list
        jobs_list.append({
            "file": new_filename,
            "file_id": batch_input_file.id,
            "job_id": new_batch_job.id,
            "status": new_batch_job.status,
            "created_at": new_batch_job.created_at,
            "is_retry": True,
            "parent_job_id": job_info["job_id"]
        })
    
    # Mark original as retried
    job_info["status"] = "failed_retried"
            
if __name__ == "__main__":
    BATCH_DIR = Path("./data/processed_experiment/batch_input2")
    process_batch_results(BATCH_DIR)
