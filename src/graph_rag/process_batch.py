import json
import os
import networkx as nx
from pathlib import Path
from openai import OpenAI
from src.config import DATA_DIR

BATCH_DIR = DATA_DIR / "graph_rag" / "batch_jobs"
STATUS_FILE = BATCH_DIR / "active_batches.json"
GRAPH_FILE = DATA_DIR / "graph_rag" / "knowledge_graph.gexf"

def process_batches():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if not STATUS_FILE.exists():
        print("No active batches found.")
        return

    with open(STATUS_FILE, "r") as f:
        active_batches = json.load(f)

    all_completed = True
    graph_updates = []

    print("\n--- Batch Status Check ---")
    for filename, info in active_batches.items():
        batch_id = info['batch_id']
        current_status = info['status']
        
        if current_status == 'processed':
            print(f"[{filename}] Already Processed.")
            continue

        try:
            # Check Status
            batch = client.batches.retrieve(batch_id)
            new_status = batch.status
            print(f"[{filename}] Status: {new_status}")
            
            # Update Status in file
            active_batches[filename]['status'] = new_status
            
            if new_status == 'completed':
                output_file_id = batch.output_file_id
                if output_file_id:
                    print(f"  - Downloading results ({output_file_id})...")
                    content = client.files.content(output_file_id).text
                    
                    # Save raw output for backup
                    output_path = BATCH_DIR / f"output_{filename}"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    # Parse and Collect Graph Data
                    if "community_batch" in filename:
                        print("  - Parsing summaries...")
                        summaries = parse_community_output(content)
                        save_communities(summaries)
                        print(f"  - Extracted {len(summaries)} summaries.")
                    else:
                        print("  - Parsing results...")
                        updates = parse_batch_output(content)
                        graph_updates.extend(updates)
                        print(f"  - Extracted {len(updates)} graph elements.")
                    
                    active_batches[filename]['status'] = 'processed'
                else:
                    print("  - Warning: Completed but no output file ID.")
            
            elif new_status in ['failed', 'expired', 'cancelled']:
                print(f"  - Job Failed/Cancelled. Check OpenAI Dashboard.")
                if batch.errors:
                    print(f"  - Errors: {batch.errors}")
            
            else:
                all_completed = False
                
        except Exception as e:
            print(f"  - Error checking {batch_id}: {e}")
            all_completed = False

    # Save Status Updates
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(active_batches, f, indent=2)

    # Build and Save Graph if we have new data
    if graph_updates:
        update_graph(graph_updates)

    if all_completed:
        print("\nAll batches finished processing!")
    else:
        print("\nSome batches are still in progress. Run this script again later.")

def parse_community_output(jsonl_content):
    """Parses output for community summaries (Text content)."""
    summaries = {}
    for line in jsonl_content.splitlines():
        if not line.strip(): continue
        try:
            data = json.loads(line)
            custom_id = data.get('custom_id', '') # e.g., community_12
            
            response = data.get('response', {})
            if response.get('status_code') != 200: continue
            
            body = response.get('body', {})
            choices = body.get('choices', [])
            if not choices: continue
            
            content = choices[0]['message']['content']
            
            # extract ID from custom_id="community_12"
            if custom_id.startswith("community_"):
                c_id = custom_id.split("_")[1]
                summaries[c_id] = content
                
        except Exception:
            pass
    return summaries

def save_communities(new_summaries):
    """Update community_summaries.json"""
    summary_file = DATA_DIR / "graph_rag" / "community_summaries.json"
    
    existing = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            try: existing = json.load(f)
            except: pass
            
    existing.update(new_summaries)
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"Updated {summary_file}")

def parse_batch_output(jsonl_content):
    """Parses the OpenAI Batch Output JSONL to extract function arguments."""
    updates = []
    for line in jsonl_content.splitlines():
        if not line.strip(): continue
        try:
            data = json.loads(line)
            # OpenAI Batch Output Structure:
            # { "id": "...", "custom_id": "...", "response": { "status_code": 200, "body": { ... "choices": [...] } } }
            
            response = data.get('response', {})
            if response.get('status_code') != 200:
                print(f"Error in response for {data.get('custom_id')}: {response.get('status_code')}")
                continue
                
            body = response.get('body', {})
            choices = body.get('choices', [])
            if not choices: continue
            
            message = choices[0].get('message', {})
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                for tool in tool_calls:
                    if tool['function']['name'] == 'GraphExtraction':
                        args = json.loads(tool['function']['arguments'])
                        updates.append(args)
        except Exception as e:
            print(f"Error parsing line: {e}")
            
    return updates

def update_graph(graph_data_list):
    """Updates the NetworkX graph with extracted data."""
    print(f"\nUpdating Graph with {len(graph_data_list)} new extractions...")
    
    # Load existing or create new
    if GRAPH_FILE.exists():
        try:
            G = nx.read_gexf(GRAPH_FILE)
            print(f"Loaded existing graph: {G.number_of_nodes()} nodes.")
        except Exception as e:
            print(f"Warning: Could not read existing graph ({e}). Creating new graph.")
            G = nx.Graph()
    else:
        G = nx.Graph()
        print("Created new graph.")
    
    for item in graph_data_list:
        # Entities
        for Entity in item.get('entities', []):
            name = Entity.get('name')
            if not name: continue
            
            # Sanitize None values
            desc = Entity.get('description', '') or ""
            ent_type = Entity.get('type', 'UNKNOWN') or "UNKNOWN"
            
            if not G.has_node(name):
                G.add_node(
                    name, 
                    type=ent_type, 
                    description=desc
                )
        
        # Relationships
        for Rel in item.get('relationships', []):
            src = Rel.get('source')
            tgt = Rel.get('target')
            if not src or not tgt: continue
            
            # Sanitize None values
            rel_desc = Rel.get('description', '') or ""
            rel_type = Rel.get('relation_type', 'RELATED') or "RELATED"

            # Ensure nodes exist (fallback)
            if not G.has_node(src): G.add_node(src, type='UNKNOWN', description="")
            if not G.has_node(tgt): G.add_node(tgt, type='UNKNOWN', description="")
            
            G.add_edge(
                src, tgt, 
                relation=rel_type,
                description=rel_desc
            )
            
    # Save
    nx.write_gexf(G, GRAPH_FILE)
    print(f"Graph Saved to {GRAPH_FILE}")
    print(f"Total: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")

def check_status_only():
    """Reads active batches and checks their status without processing."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if not STATUS_FILE.exists():
        print("No active batches found.")
        return

    with open(STATUS_FILE, "r") as f:
        active_batches = json.load(f)

    print(f"\n{'Batch File':<50} | {'Status':<15} | {'Batch ID'}")
    print("-" * 100)

    for filename, info in active_batches.items():
        batch_id = info['batch_id']
        try:
            batch = client.batches.retrieve(batch_id)
            print(f"{filename:<50} | {batch.status:<15} | {batch_id}")
            
            # Show error snippet if failed
            if batch.status == 'failed' and batch.errors:
                print(f"  >>> Errors: {batch.errors}")
                
        except Exception as e:
            print(f"{filename:<50} | Error: {e}")
    print("-" * 100)

def rebuild_from_cache():
    """Rebuilds the graph using locally saved output files (recovery mode)."""
    print("\n--- Rebuilding Graph from Local Cache ---")
    
    output_files = list(BATCH_DIR.glob("output_merged_batch_requests_part_*.jsonl"))
    if not output_files:
        print(f"No output files found in {BATCH_DIR}. Cannot rebuild.")
        return
        
    all_updates = []
    for out_file in output_files:
        print(f"Reading {out_file.name}...")
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
            updates = parse_batch_output(content)
            all_updates.extend(updates)
            print(f"  - Extracted {len(updates)} elements.")
        except Exception as e:
            print(f"  - Error reading file: {e}")
            
    if all_updates:
        update_graph(all_updates)
    else:
        print("No data extracted from files.")

if __name__ == "__main__":
    import sys
    # python -m src.graph_rag.process_batch check
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_status_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_from_cache()
    else:
        process_batches()
