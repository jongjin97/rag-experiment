import json
import os
import networkx as nx
import tiktoken
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from src.config import DATA_DIR, MODEL_NAME

# Config
# User confirmed 1M context, but we keep a sane limit to avoid noise.
# Top 1000 nodes ~ 80k-100k tokens.
MAX_NODES_PER_COMMUNITY = 2000 
TOKENS_PER_FILE_LIMIT = 1_500_000

GRAPH_FILE = DATA_DIR / "graph_rag" / "knowledge_graph.gexf"
BATCH_DIR = DATA_DIR / "graph_rag" / "batch_jobs"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_TEMPLATE = """You are an expert business analyst reviewing a knowledge graph community from Samsung Business Reports.

Summarize the following community of entities and relationships. 
Focus on:
1. What is the central theme of this community? (e.g., "Semiconductor Market Trend", "Mobile Division Performance")
2. Key entities and their roles.
3. Important relationships and events.
4. Any financial figures or strategic decisions mentioned.

Community Data:
{community_data}

Summary:"""

def count_tokens(text: str) -> int:
    try:
        from tiktoken import encoding_for_model
        enc = encoding_for_model("gpt-4")
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_community_subgraph(G, nodes):
    """Sorts nodes by degree centrality within the community to pick top ones."""
    subgraph = G.subgraph(nodes)
    
    if len(nodes) <= MAX_NODES_PER_COMMUNITY:
        return nodes
        
    # Calculate degree inside the subgraph
    degrees = dict(subgraph.degree())
    # Sort by degree desc
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    # Pick top N
    top_nodes = [n for n, d in sorted_nodes[:MAX_NODES_PER_COMMUNITY]]
    return top_nodes

def format_community_text(G, c_id, nodes):
    lines = [f"Community ID: {c_id}"]
    
    # Nodes
    lines.append("\nEntities:")
    for node in nodes:
        data = G.nodes[node]
        desc = data.get('description', '') or ""
        type_ = data.get('type', 'UNKNOWN')
        lines.append(f"- {node} ({type_}): {desc}")
        
    # Internal Edges (only if both ends are in the selected nodes)
    lines.append("\nRelationships:")
    sub_G = G.subgraph(nodes)
    for u, v, data in sub_G.edges(data=True):
        rel = data.get('relation', 'RELATED')
        desc = data.get('description', '') or ""
        lines.append(f"- {u} -> {rel} -> {v}: {desc}")
        
    return "\n".join(lines)

def prepare_community_batch():
    if not GRAPH_FILE.exists():
        print("Graph file missing.")
        return

    print(f"Loading Graph from {GRAPH_FILE}...")
    G = nx.read_gexf(GRAPH_FILE)
    print(f"Loaded {G.number_of_nodes()} nodes.")

    # Group by Community
    communities = defaultdict(list)
    for node, data in G.nodes(data=True):
        c_id = data.get('community')
        if c_id is not None:
            communities[str(c_id)].append(node)
            
    print(f"Found {len(communities)} communities.")

    files_created = []
    current_requests = []
    current_tokens = 0
    file_index = 1

    sorted_community_ids = sorted(communities.keys(), key=lambda k: len(communities[k]), reverse=True)

    for c_id in tqdm(sorted_community_ids, desc="Preparing Requests"):
        all_nodes = communities[c_id]
        
        # Filter (Top N)
        selected_nodes = get_community_subgraph(G, all_nodes)
        
        # Format
        text_data = format_community_text(G, c_id, selected_nodes)
        prompt_content = SUMMARY_TEMPLATE.replace("{community_data}", text_data)
        
        tokens_est = count_tokens(prompt_content) + 200
        
        # Batch Request Body
        request_body = {
            "custom_id": f"community_{c_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
                "temperature": 0.0,
                "max_tokens": 2000
            }
        }
        
        # Check File Limit
        if current_tokens + tokens_est > TOKENS_PER_FILE_LIMIT:
            save_batch_file(current_requests, file_index)
            files_created.append(file_index)
            file_index += 1
            current_requests = []
            current_tokens = 0
            
        current_requests.append(request_body)
        current_tokens += tokens_est

    # Save Last
    if current_requests:
        save_batch_file(current_requests, file_index)
        files_created.append(file_index)

    print(f"\nSuccessfully created {len(files_created)} community batch files.")

def save_batch_file(requests, index):
    filename = BATCH_DIR / f"community_batch_part_{index}.jsonl"
    print(f"Writing {len(requests)} requests to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    prepare_community_batch()
