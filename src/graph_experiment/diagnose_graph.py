
import networkx as nx
from pathlib import Path
import sys

# Add project root to path to import config if needed, though we can just use relative paths for now
# from src.config import DATA_DIR

# Define Path Manually or Import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_FILE = BASE_DIR / "data" / "processed_experiment" / "batch_input" / "graph.gexf"

def diagnose_graph_health():
    print(f"Loading Graph from {GRAPH_FILE}...")
    if not GRAPH_FILE.exists():
        print("Graph file not found!")
        return

    G = nx.read_gexf(GRAPH_FILE)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    print(f"\n[Basic Stats]")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    if num_nodes > 0:
        print(f"Average Degree: {2 * num_edges / num_nodes:.2f}") # Undirected estimation

    # 1. Connected Components
    # For directed graph, we usually check Weakly Connected Components for reachability
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
        print(f"\n[1. Connected Components] (Weakly Connected)")
    else:
        components = list(nx.connected_components(G))
        print(f"\n[1. Connected Components]")
        
    num_components = len(components)
    print(f"Number of Components (Islands): {num_components}")
    print(f"Ideal: 1 (or < 10 for very large graphs)")
    
    # 2. Giant Component Fraction
    if num_nodes > 0:
        largest_cc = max(components, key=len)
        size_largest = len(largest_cc)
        giant_fraction = (size_largest / num_nodes) * 100
        print(f"\n[2. Giant Component Fraction]")
        print(f"Largest Component Size: {size_largest}")
        print(f"Fraction: {giant_fraction:.2f}%")
        print(f"Target: > 80%")
    
    # 3. Isolates & Leaves
    degrees = dict(G.degree())
    isolates = [n for n, d in degrees.items() if d == 0]
    leaves = [n for n, d in degrees.items() if d == 1]
    
    iso_fraction = (len(isolates) / num_nodes) * 100 if num_nodes else 0
    leaf_fraction = (len(leaves) / num_nodes) * 100 if num_nodes else 0
    
    print(f"\n[3. Isolates & Leaves]")
    print(f"Isolates (Degree 0): {len(isolates)} ({iso_fraction:.2f}%)")
    print(f"Leaves (Degree 1): {len(leaves)} ({leaf_fraction:.2f}%)")
    print(f"Target: Isolates ~ 0%, Leaves < 20-30%")
    
    # Diagnosis
    print("\n[Diagnosis]")
    if giant_fraction < 10:
        print("CRITICAL: Graph is shattered. 'Giant Component' is virtually non-existent.")
        print("Reason: Entity resolution failed (synonyms not merged) or Extraction density too low.")
    elif giant_fraction < 50:
        print("WARNING: Graph is fragmented. Information flow is blocked between major clusters.")
    else:
        print("GOOD: Graph has a healthy connective core.")
        
    if iso_fraction > 10:
        print("WARNING: High number of isolated nodes. Check extraction logic (orphan entities).")

if __name__ == "__main__":
    diagnose_graph_health()
