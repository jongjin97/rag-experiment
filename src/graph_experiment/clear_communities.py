
import networkx as nx
from pathlib import Path
import sys

# Define Path Manually
BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_FILE = BASE_DIR / "data" / "processed_experiment" / "batch_input2" / "graph.gexf"

def clear_communities():
    print(f"Loading Graph from {GRAPH_FILE}...")
    if not GRAPH_FILE.exists():
        print("Graph file not found!")
        return

    G = nx.read_gexf(GRAPH_FILE)
    print(f"Graph Loaded: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")
    
    # Attributes to remove
    # We look for any attribute containing 'community'
    # GEXF extraction puts attributes in nodes.
    
    count = 0
    keys_to_remove = set()
    
    # First pass: Identify keys to remove from a sample or check all
    # Since nodes might have different attributes, we check all.
    
    print("Scanning and removing community attributes...")
    for node, data in G.nodes(data=True):
        # Identify keys to delete for this node
        node_keys = [k for k in data.keys() if "community" in k]
        
        if node_keys:
            count += 1
            for k in node_keys:
                del data[k]
                keys_to_remove.add(k)
                
    print(f"Removed community attributes from {count} nodes.")
    print(f"Attributes removed: {keys_to_remove}")
    
    print(f"Saving cleaned graph to {GRAPH_FILE}...")
    nx.write_gexf(G, GRAPH_FILE)
    print("Done.")

if __name__ == "__main__":
    clear_communities()
