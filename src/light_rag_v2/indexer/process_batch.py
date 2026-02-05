import json
import asyncio
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List

from src.config import LIGHT_RAG_V2_DIR, OPENAI_API_KEY
from src.light_rag_v2.storage.graph import GraphStorage
from src.light_rag_v2.storage.vector import VectorStorage
from src.light_rag_v2.utils.embedding import get_embedding_function

INDEXER_DIR = LIGHT_RAG_V2_DIR / "indexer"
CHUNKS_FILE = INDEXER_DIR / "chunks.json"
BATCH_OUTPUT_FILE = INDEXER_DIR / "batch_output.jsonl" # This will be the result from OpenAI

async def process_batch_results():
    """
    1. Load chunks.json
    2. Load batch_output.jsonl
    3. Build Graph
    4. Populate Vector DBs (Nodes, Relations, Chunks)
    """
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found. Run prepare_batch.py first.")
        return

    print("Loading chunks map...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    print("Loading batch output...")
    if not BATCH_OUTPUT_FILE.exists():
        print(f"Error: {BATCH_OUTPUT_FILE} not found. Please provide the batch result file.")
        return
        
    # Initialize Storages
    graph_storage = GraphStorage()
    entity_vec = VectorStorage("light_rag_v2_entities")
    relation_vec = VectorStorage("light_rag_v2_relations")
    chunk_vec = VectorStorage("light_rag_v2_chunks")
    
    embedding_fn = get_embedding_function()
    
    # Graphs
    G = nx.Graph()
    
    # Temp storage before creating vectors
    nodes_data = {} # id -> {type, desc, communities, chunks}
    edges_data = {} # id -> {source, target, relation, desc, chunks}
    
    print("Parsing batch results...")
    with open(BATCH_OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                res = json.loads(line)
                custom_id = res.get("custom_id") # chunk_id
                
                # Check for successful response
                response = res.get("response", {})
                if response.get("status_code") != 200:
                    print(f"Skipping failed request {custom_id}")
                    continue
                    
                body = response.get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    continue
                    
                tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                if not tool_calls:
                    continue
                    
                for tool in tool_calls:
                    if tool["function"]["name"] == "GraphExtraction":
                        try:
                            args = json.loads(tool["function"]["arguments"])
                        except json.JSONDecodeError:
                            continue
                        
                        # Process Entities
                        for ent in args.get("entities", []):
                            name = ent["name"]
                            if name not in nodes_data:
                                nodes_data[name] = {
                                    "type": ent.get("type") or "",
                                    "description": ent.get("description") or "",
                                    "source_chunk_ids": set()
                                }
                            nodes_data[name]["source_chunk_ids"].add(str(custom_id))
                        
                        # Process Relationships
                        for rel in args.get("relationships", []):
                            src = rel["source"]
                            tgt = rel["target"]
                            rel_type = rel["relation_type"]
                            # Edge ID convention
                            edge_id = f"{src}-{rel_type}-{tgt}"
                            
                            if edge_id not in edges_data:
                                edges_data[edge_id] = {
                                    "source": src,
                                    "target": tgt,
                                    "relation": rel_type,
                                    "description": rel.get("description") or "",
                                    "source_chunk_ids": set()
                                }
                            edges_data[edge_id]["source_chunk_ids"].add(str(custom_id))
                        
                        # Fix NoneTypes during vector metadata creation
                        # (Will happen in next steps, but good to ensure dictionary is clean)
            except json.JSONDecodeError:
                # Silently skip invalid lines
                continue
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue

    # 1. Build & Save Graph
    print(f"Building Graph with {len(nodes_data)} nodes and {len(edges_data)} edges...")
    for node, data in nodes_data.items():
        G.add_node(node, 
                   type=data["type"],
                   description=data["description"],
                   source_chunk_ids=json.dumps(list(data["source_chunk_ids"])))
        
    for eid, data in edges_data.items():
        G.add_edge(data["source"], data["target"], 
                   relation=data["relation"], 
                   description=data["description"],
                   source_chunk_ids=json.dumps(list(data["source_chunk_ids"])))
    
    # Save GEXF
    # For now, saving to a new path in light_rag dir to avoid overwriting original without backup
    new_graph_path = LIGHT_RAG_V2_DIR / "lightrag_graph.gexf"
    nx.write_gexf(G, str(new_graph_path))
    print(f"Graph saved to {new_graph_path}")

    # 2. Populate Entity Vector DB
    print("Vectorizing Nodes...")
    node_ids = []
    node_texts = []
    node_metas = []
    
    for node, data in nodes_data.items():
        text = f"{node} ({data['type']}): {data['description']}"
        node_ids.append(node)
        node_texts.append(text)
        node_metas.append({
            "entity_name": node,
            "type": data['type'],
            "description": data['description']
        })
        
    if node_texts:
        embeddings = embedding_fn.embed_documents(node_texts)
        for i, nid in enumerate(tqdm(node_ids, desc="Upserting Nodes")):
             await entity_vec.upsert(nid, embeddings[i], node_metas[i])

    # 3. Populate Relation Vector DB
    print("Vectorizing Edges...")
    edge_ids = []
    edge_texts = []
    edge_metas = []
    
    for eid, data in edges_data.items():
        text = f"{data['source']} -[{data['relation']}]-> {data['target']}: {data['description']}"
        edge_ids.append(eid)
        edge_texts.append(text)
        edge_metas.append({
            "source": data['source'],
            "target": data['target'],
            "relation": data['relation'],
            "description": data['description']
        })

    if edge_texts:
        embeddings = embedding_fn.embed_documents(edge_texts)
        for i, eid in enumerate(tqdm(edge_ids, desc="Upserting Edges")):
            await relation_vec.upsert(eid, embeddings[i], edge_metas[i])

    # 4. Populate Chunk Vector DB
    print("Vectorizing Original Chunks...")
    chunk_ids = []
    chunk_texts = []
    chunk_metas = []
    
    for cid, data in chunk_map.items():
        chunk_ids.append(cid)
        chunk_texts.append(data["content"])
        chunk_metas.append({
            "source": data["source"],
            "page": data["page"]
        })
        
    if chunk_texts:
        # Processing in batches for embedding model safety
        batch_size = 32
        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding Chunks"):
            batch_texts = chunk_texts[i:i+batch_size]
            batch_ids = chunk_ids[i:i+batch_size]
            batch_metas = chunk_metas[i:i+batch_size]
            
            embeddings = embedding_fn.embed_documents(batch_texts)
            
            for j, cid in enumerate(batch_ids):
                await chunk_vec.upsert(cid, embeddings[j], batch_metas[j])

    print("Processing Complete!")

if __name__ == "__main__":
    asyncio.run(process_batch_results())
