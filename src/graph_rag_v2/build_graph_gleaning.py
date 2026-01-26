
import asyncio
import os
import networkx as nx
from tqdm.asyncio import tqdm
from pathlib import Path

from src.utils.document_loader import load_documents_merged
from src.rag_best_practices.chunking import split_documents
from src.config import DATA_DIR
from src.graph_rag_v2.extractor_gleaning import GleaningGraphExtractor, GraphExtraction

# Paths
DOC_PATH = DATA_DIR / "samsung"
GRAPH_FILE = DATA_DIR / "graph_rag_v2" / "knowledge_graph_gleaning.gexf"
GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)

async def build_graph_async(limit: int = None):
    print(f"Loading documents from {DOC_PATH}...")
    docs = load_documents_merged(str(DOC_PATH))
    chunks = split_documents(docs, chunk_size=1024, chunk_overlap=100)
    
    if limit:
        print(f"Selecting top {limit} chunks for testing...")
        chunks = chunks[:limit]
    
    print(f"Total Chunks to Process: {len(chunks)}")
    
    extractor = GleaningGraphExtractor(max_gleanings=3)
    semaphore = asyncio.Semaphore(5) # Limit concurrency
    
    results = []

    async def sem_extract(chunk_text):
        async with semaphore:
            return await extractor.extract_async(chunk_text)

    # Run extraction
    tasks = [sem_extract(chunk.page_content) for chunk in chunks]
    
    # Use tqdm for progress bar
    results = await tqdm.gather(*tasks, desc="Extracting (Gleaning)")
    
    # Build Graph
    update_graph_local(results)

def update_graph_local(extractions: list[GraphExtraction]):
    print(f"\nBuilding Graph from {len(extractions)} extractions...")
    
    if GRAPH_FILE.exists():
        try:
            G = nx.read_gexf(GRAPH_FILE)
            print(f"Loaded existing graph: {G.number_of_nodes()} nodes.")
        except Exception:
            G = nx.Graph()
    else:
        G = nx.Graph()
        
    for item in extractions:
        # Pydantic objects, not dicts
        for entity in item.entities:
            # entity.name, entity.type, entity.description
            if not G.has_node(entity.name):
                G.add_node(
                    entity.name,
                    type=entity.type,
                    description=entity.description
                )
        
        for rel in item.relationships:
            # rel.source, rel.target, rel.relation_type, rel.description
            if not G.has_node(rel.source):
                G.add_node(rel.source, type='UNKNOWN', description="")
            if not G.has_node(rel.target):
                G.add_node(rel.target, type='UNKNOWN', description="")
                
            G.add_edge(
                rel.source, rel.target,
                relation=rel.relation_type,
                description=rel.description or ""
            )

    nx.write_gexf(G, GRAPH_FILE)
    print(f"Graph Saved to {GRAPH_FILE}")
    print(f"Total: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of chunks for testing")
    args = parser.parse_args()
    
    asyncio.run(build_graph_async(limit=args.limit))
