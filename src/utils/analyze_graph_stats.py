import networkx as nx
from pathlib import Path
from src.config import DATA_DIR

def analyze_graphs():
    v1_path = DATA_DIR / "graph_rag" / "knowledge_graph.gexf"
    v2_path = DATA_DIR / "graph_rag_v2" / "knowledge_graph.gexf"

    print("--- Graph Structural Analysis ---\n")

    # Analyze v1
    if v1_path.exists():
        try:
            g1 = nx.read_gexf(v1_path)
            print(f"[Graph v1 (English Prompt / Top 10 Limit)]")
            print(f"- Path: {v1_path}")
            print(f"- Nodes: {g1.number_of_nodes()}")
            print(f"- Edges: {g1.number_of_edges()}")
            if g1.number_of_nodes() > 0:
                print(f"- Density: {nx.density(g1):.6f}")
            print("")
        except Exception as e:
            print(f"[Graph v1] Error reading file: {e}\n")
    else:
        print(f"[Graph v1] File not found at {v1_path}\n")

    # Analyze v2
    if v2_path.exists():
        try:
            g2 = nx.read_gexf(v2_path)
            print(f"[Graph v2 (Korean Prompt / No Limit)]")
            print(f"- Path: {v2_path}")
            print(f"- Nodes: {g2.number_of_nodes()}")
            print(f"- Edges: {g2.number_of_edges()}")
            if g2.number_of_nodes() > 0:
                print(f"- Density: {nx.density(g2):.6f}")
            print("")
        except Exception as e:
            print(f"[Graph v2] Error reading file: {e}\n")
    else:
        print(f"[Graph v2] File not found at {v2_path}\n")

if __name__ == "__main__":
    analyze_graphs()
