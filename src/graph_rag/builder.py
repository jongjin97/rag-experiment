import networkx as nx
import pickle
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
from networkx.algorithms.community import greedy_modularity_communities
from src.utils.document_loader import load_documents
from src.rag_best_practices.retrieval import split_documents
from src.config import DATA_DIR
from src.graph_rag.extractor import GraphExtractor
from src.config import DATA_DIR
import leidenalg
import igraph as ig
GRAPH_DIR = DATA_DIR / "graph_rag"
GRAPH_FILE = GRAPH_DIR / "knowledge_graph.gexf"
COMMUNITY_FILE = GRAPH_DIR / "community_map.pkl"

class GraphBuilder:
    def __init__(self):
        self.extractor = GraphExtractor()
        self.graph = nx.Graph()
        
    async def process_chunk(self, chunk: str, semaphore: asyncio.Semaphore):
        """
        Process a single chunk asynchronously with semaphore limit.
        """
        async with semaphore:
            result = await self.extractor.extract_async(chunk)
            return result

    async def build_from_chunks_async(self, chunks: List[str], max_concurrency: int = 10):
        """
        Builds graph asynchronously from chunks.
        """
        print(f"Building Graph from {len(chunks)} chunks (Async, Max Concurrency: {max_concurrency})...")
        
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [self.process_chunk(chunk, semaphore) for chunk in chunks]
        
        print(f"Tasks: {tasks}")
        # Process in batches or gather all (gather all is fine with semaphore)
        # We use tqdm for progress bar
        
        results = []
        # Use as_completed to update graph incrementally (better for memory/saving?)
        # For simplicity and speed, gather is easiest, but let's use tqdm
        
        for f in tqdm(asyncio.as_completed(tasks), total=len(chunks), desc="Async Extraction"):
            result = await f
            print(f"Result: {result}")
            # --- Update Graph (Not Thread-Safe? NetworkX is not thread safe, but asyncio is single threaded event loop)
            # Since we are in asyncio (single thread), direct graph update is safe!
            
            # Add Entities (Nodes)
            for entity in result.entities:
                if not self.graph.has_node(entity.name):
                    self.graph.add_node(
                        entity.name, 
                        type=entity.type, 
                        description=entity.description
                    )
            
            # Add Relationships (Edges)
            for rel in result.relationships:
                if not self.graph.has_node(rel.source):
                    self.graph.add_node(rel.source, type="UNKNOWN", description="")
                if not self.graph.has_node(rel.target):
                    self.graph.add_node(rel.target, type="UNKNOWN", description="")
                
                self.graph.add_edge(
                    rel.source, 
                    rel.target, 
                    relation=rel.relation_type,
                    description=rel.description
                )
                
        print(f"Graph Built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges.")

    def detect_communities(self):
        """
        Detects communities using Leiden Algorithm (via leidenalg & igraph).
        Benefits: Better modularity, no disconnected communities, faster.
        """
        print("Detecting Communities using Leiden Algorithm...")
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, skipping community detection.")
            return {}

        # communities = greedy_modularity_communities(self.graph)

        # Convert NetworkX graph to iGraph
        # Mapping node names to indices
        node_names = list(self.graph.nodes())
        node_map = {name: i for i, name in enumerate(node_names)}
        edges = []
        for u, v in self.graph.edges():
            if u in node_map and v in node_map:
                edges.append((node_map[u], node_map[v]))
        
        ig_graph = ig.Graph(n=len(node_names), edges=edges, directed=False)
        
        # Apply Leiden (ModularityVertexPartition)
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
        
        # Map back to NetworkX nodes
        community_map = {}
        communities = []
        
        for i, cluster_nodes in enumerate(partition):
            # cluster_nodes is list of node indices
            members = [node_names[idx] for idx in cluster_nodes]
            communities.append(frozenset(members))
        print(f"Communities: {communities}")
        
        print(f"Detected {len(communities)} communities (Leiden).")
        return self._apply_communities(communities)

    def _apply_communities(self, communities):
        """Helper to apply community IDs to graph and return map."""
        community_map = {}
        for c_id, community_set in enumerate(communities):
            for node in community_set:
                self.graph.nodes[node]['community'] = c_id
                community_map[node] = c_id
                
        print(f"Community Map: {community_map}")
        print(f"Detected {len(communities)} communities.")
        return community_map

    def save_graph(self):
        """Saves the graph and community map."""
        if not GRAPH_DIR.exists():
            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            
        nx.write_gexf(self.graph, GRAPH_FILE)
        
        community_data = {node: data.get('community', -1) for node, data in self.graph.nodes(data=True)}
        with open(COMMUNITY_FILE, "wb") as f:
            pickle.dump(community_data, f)
            
        print(f"Graph saved to {GRAPH_FILE}")
        print(f"Community map saved to {COMMUNITY_FILE}")

async def main():
    # Test Block
    
    DOC_PATH = DATA_DIR / "samsung" 
    
    print("Loading Document...")
    docs = load_documents(DOC_PATH) 
    # Use 1024 token chunks for Graph to capture more context per node
    chunks = split_documents(docs, chunk_size=1024, chunk_overlap=100)
    
    # Process ALL chunks (or subset if needed)
    # Using 200 limit as requested by user initially, or full? 
    # User said "4240 chunks... trying to reduce to 200", then asked "can we reduce time for 4240?"
    # Answer: YES. So let's aim for ALL 4240 or a large subset (e.g. 200) as a demo.
    # Let's do 200 for verified speedup, then user can run full.
    test_chunks = [c.page_content for c in chunks[:1]] 
    # test_chunks = [c.page_content for c in chunks] # Uncomment for full run
    
    builder = GraphBuilder()
    await builder.build_from_chunks_async(test_chunks, max_concurrency=5) # 10 concurrent requests
    builder.detect_communities()
    builder.save_graph()

if __name__ == "__main__":
    asyncio.run(main())
