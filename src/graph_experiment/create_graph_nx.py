
import json
import os
import networkx as nx
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY

def ensure_batch_results_downloaded(batch_dir: Path):
    """
    Checks submitted_jobs.json and downloads missing result files.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    record_file = batch_dir / "submitted_jobs.json"
    
    if not record_file.exists():
        print("No job records found.")
        return []

    with open(record_file, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    downloaded_files = []
    
    print(f"Checking {len(jobs)} jobs for results...")
    
    for job in jobs:
        job_id = job["job_id"]
        output_filename = f"batch_output_{job_id}.jsonl"
        output_path = batch_dir / output_filename
        
        if output_path.exists():
            downloaded_files.append(output_path)
            continue
            
        # If missing, check status and download
        try:
            batch_job = client.batches.retrieve(job_id)
            if batch_job.status == "completed" and batch_job.output_file_id:
                print(f"Downloading results for {job_id}...")
                content = client.files.content(batch_job.output_file_id).content
                with open(output_path, "wb") as f:
                    f.write(content)
                print(f"  - Saved to {output_filename}")
                downloaded_files.append(output_path)
            else:
                print(f"Job {job_id} is {batch_job.status} (No output yet)")
        except Exception as e:
            print(f"Error checking job {job_id}: {e}")
            
    return downloaded_files

def create_graph_nx(batch_dir: Path):
    # 1. Ensure data availablity
    result_files = ensure_batch_results_downloaded(batch_dir)
    
    if not result_files:
        print("No result files available to build graph.")
        return

    # 2. Initialize NetworkX Graph
    G = nx.Graph()
    
    print(f"\nBuilding Graph from {len(result_files)} files...")
    
    total_entities = 0
    total_relationships = 0
    
    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                
                try:
                    # Parse Batch Response
                    response = json.loads(line)
                    if "response" not in response or not response["response"]:
                        continue
                        
                    body = response["response"]["body"]
                    if not body: continue
                    
                    message = body["choices"][0]["message"]
                    
                    # NEW: Handle Tool Calls (for graph_rag_v2 style)
                    graph_data = None
                    
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_call = message["tool_calls"][0]
                        function_args = tool_call["function"]["arguments"]
                        graph_data = json.loads(function_args)
                    else:
                        # Fallback for old content-based or markdown format
                        content_str = message.get("content", "")
                        if content_str:
                             # Clean Markdown Code Blocks (```json ... ```)
                            if content_str.strip().startswith("```"):
                                content_str = content_str.strip().strip("`").replace("json", "", 1).strip()
                            try:
                                graph_data = json.loads(content_str)
                            except:
                                pass

                    if not graph_data:
                        continue
                    
                    # Add Entities as Nodes
                    for entity in graph_data.get("entities", []):
                        # GEXF does not support NoneType for attributes
                        desc = entity.get("description", "")
                        if desc is None: desc = ""
                        
                        G.add_node(entity["name"], type=entity["type"], description=desc)
                        total_entities += 1
                        
                    # Add Relationships as Edges
                    for rel in graph_data.get("relationships", []):
                        # GEXF does not support NoneType for attributes
                        desc = rel.get("description", "")
                        if desc is None: desc = ""
                        
                        G.add_edge(
                            rel["source"], 
                            rel["target"], 
                            relation=rel["relation_type"], 
                            description=desc
                        )
                        total_relationships += 1
                        
                except json.JSONDecodeError as e:
                     print(f"Error parsing JSON in {file_path.name}: {e}")
                except Exception as e:
                    print(f"Error processing line in {file_path.name}: {e}")

    print(f"\nGraph Construction Complete:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    
    # 3. Save Graph
    output_gexf = batch_dir / "graph.gexf"
    nx.write_gexf(G, output_gexf)
    print(f"Saved graph to {output_gexf}")
    
    output_graphml = batch_dir / "graph.graphml"
    nx.write_graphml(G, output_graphml)
    print(f"Saved graph to {output_graphml}")

if __name__ == "__main__":
    BATCH_DIR = Path("./data/processed_experiment/batch_input")
    create_graph_nx(BATCH_DIR)
