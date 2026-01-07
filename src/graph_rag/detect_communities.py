import networkx as nx
import os
try:
    import leidenalg
    import igraph as ig
except ImportError:
    print("Error: leidenalg or igraph not installed.")
    print("Please run: pip install leidenalg igraph")
    exit(1)

from src.config import DATA_DIR

GRAPH_FILE = DATA_DIR / "graph_rag" / "knowledge_graph.gexf"

def detect_communities_standalone():
    print(f"Loading Graph from {GRAPH_FILE}...")
    if not GRAPH_FILE.exists():
        print("Graph file not found!")
        return

    G = nx.read_gexf(GRAPH_FILE)
    print(f"Graph Loaded: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")

    print("Converting to iGraph for Leiden Algorithm...")
    # Map node names to indices
    node_names = list(G.nodes())
    node_map = {name: i for i, name in enumerate(node_names)}
    edges = []
    for u, v in G.edges():
        if u in node_map and v in node_map:
            edges.append((node_map[u], node_map[v]))
    
    ig_graph = ig.Graph(n=len(node_names), edges=edges, directed=False)
    
    print("Running Leiden Algorithm...")
    # Partition type: ModularityVertexPartition is standard for community detection
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    
    print(f"Detected {len(partition)} communities.")
    
    # Map back to NetworkX
    print("Applying Community IDs to Graph...")
    community_counts = {}
    
    for c_id, cluster_nodes in enumerate(partition):
        # cluster_nodes is list of node indices
        count = 0
        for node_idx in cluster_nodes:
            node_name = node_names[node_idx]
            G.nodes[node_name]['community'] = c_id
            count += 1
        community_counts[c_id] = count
        
    # Stats
    sorted_communities = sorted(community_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Largest Communities:")
    for c_id, count in sorted_communities[:5]:
        print(f"  - Community {c_id}: {count} nodes")

    print(f"\nSaving updated graph to {GRAPH_FILE}...")
    nx.write_gexf(G, GRAPH_FILE)
    print("Done.")

if __name__ == "__main__":
    detect_communities_standalone()
