import networkx as nx
import os
from collections import defaultdict
try:
    import leidenalg
    import igraph as ig
except ImportError:
    print("Error: leidenalg or igraph not installed.")
    print("Please run: pip install leidenalg igraph")
    exit(1)

from src.config import DATA_DIR

GRAPH_FILE = DATA_DIR / "processed_experiment" / "batch_input" /"graph.gexf"

def detect_communities_standalone():
    print(f"Loading Graph from {GRAPH_FILE}...")
    if not GRAPH_FILE.exists():
        print("Graph file not found!")
        return

    G = nx.read_gexf(GRAPH_FILE)
    print(f"Graph Loaded: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")

    # --- Hierarchical Detection Setup ---
    # We want 4 levels: Level 3 (Bottom/Finest) -> Level 0 (Top/Root)
    # Strategy: Iterative Leiden on Induced Graphs
    # Constraints (Max Community Size in terms of entities or previous nodes)
    # L3: max 50 entities
    # L2: max 50 L3-nodes (approx 2500 entities)
    # L1: max 500 L2-nodes (Huge aggregations)
    # L0: No limit (Root)
    
    levels_config = [
        {"level": 3, "max_comm_size": 20}, # Bottom
        {"level": 2, "max_comm_size": 30}, 
        {"level": 1, "max_comm_size": 40},
        {"level": 0, "max_comm_size": None} # Top
    ]
    
    current_graph = G
    # Mapping from current_graph node -> original entity name (list)
    # Ideally we track: original ID -> L3 ID -> L2 ID -> L1 ID -> L0 ID
    
    # Initialize hierarchy map: node_name -> {level: comm_id}
    hierarchy = {n: {} for n in G.nodes()}
    
    # For induced graph construction
    # We need to map: node_in_current_graph -> community_id
    
    # Base Mapping: The nodes of current_graph are the entities themselves initially
    # If we induce, the nodes of next graph are the community IDs.
    
    # We need to track which original nodes belong to which current node
    # current_node_id -> [original_node_names]
    current_node_to_members = {n: [n] for n in G.nodes()}
    
    for config in levels_config:
        level = config["level"]
        max_size = config["max_comm_size"] # Max nodes in *current_graph* per community
        
        print(f"\n--- Running Level {level} Detection ---")
        print(f"  - Input Graph: {current_graph.number_of_nodes()} nodes, {current_graph.number_of_edges()} edges")
        print(f"  - Max Comm Size Constraint: {max_size if max_size else 'None'}")
        
        # 1. Convert to iGraph
        node_names = list(current_graph.nodes())
        node_map = {name: i for i, name in enumerate(node_names)}
        
        edges = []
        weights = []
        for u, v, data in current_graph.edges(data=True):
            if u in node_map and v in node_map:
                edges.append((node_map[u], node_map[v]))
                weights.append(data.get('weight', 1.0))
        
        ig_graph = ig.Graph(n=len(node_names), edges=edges, directed=False)
        if weights:
            ig_graph.es['weight'] = weights
            
        # 2. Run Leiden
        if max_size:
            partition = leidenalg.find_partition(
                ig_graph, 
                leidenalg.ModularityVertexPartition,
                weights='weight' if weights else None,
                max_comm_size=max_size,
                n_iterations=5,
                seed=42
            )
        else:
            partition = leidenalg.find_partition(
                ig_graph, 
                leidenalg.ModularityVertexPartition,
                weights='weight' if weights else None,
                n_iterations=5,
                seed=42
            )
            
        print(f"  -> Detected {len(partition)} communities.")
        
        # 3. Update Hierarchy & Prepare Next Graph
        # comm_id -> [member_original_nodes]
        next_node_to_members = defaultdict(list)
        
        # Mapping for this specific level: original_node -> comm_id
        # We need to broadcast the result to all original members
        
        for c_id, cluster_nodes in enumerate(partition):
            # cluster_nodes: indices in current_graph
            for node_idx in cluster_nodes:
                current_node_name = node_names[node_idx]
                
                # Get all original entities belonging to this current node
                original_members = current_node_to_members[current_node_name]
                
                # Assign this level's ID to them
                for member in original_members:
                    hierarchy[member][level] = c_id
                    
                # Aggregate for next level
                next_node_to_members[c_id].extend(original_members)
                
        # 4. Build Induced Graph for Next Level (if not Top)
        if level > 0:
            print("  -> Building Induced Graph for next level...")
            G_next = nx.Graph()
            next_comm_ids = list(next_node_to_members.keys())
            G_next.add_nodes_from(next_comm_ids)
            
            # Edges between communities
            # Iterate current edges, map valid ends to communities
            # current_node -> comm_id (c_id)
            current_to_comm = {}
            for c_id, cluster_nodes in enumerate(partition):
                for node_idx in cluster_nodes:
                    current_node_name = node_names[node_idx]
                    current_to_comm[current_node_name] = c_id
            
            edge_log = defaultdict(float)
            
            for u, v, data in current_graph.edges(data=True):
                w = data.get('weight', 1.0)
                c1 = current_to_comm.get(u)
                c2 = current_to_comm.get(v)
                
                if c1 is not None and c2 is not None and c1 != c2:
                    key = tuple(sorted((c1, c2)))
                    edge_log[key] += w
            
            for (c1, c2), w in edge_log.items():
                G_next.add_edge(c1, c2, weight=w)
                
            current_graph = G_next
            current_node_to_members = next_node_to_members
        else:
            print("  -> Top level reached.")

    # --- Save to Graph ---
    print("\nApplying Hierarchy to NetworkX Graph...")
    for node_name, levels in hierarchy.items():
        for level, c_id in levels.items():
            G.nodes[node_name][f'community_level_{level}'] = c_id
            # Backward compatibility for 'community' (use Level 3)
            if level == 3:
                G.nodes[node_name]['community'] = c_id

    print("\nStats:")
    for level in [3, 2, 1, 0]:
        counts = defaultdict(int)
        for n, d in G.nodes(data=True):
            cid = d.get(f'community_level_{level}')
            if cid is not None:
                counts[cid] += 1
        print(f"  Level {level} Communities: {len(counts)} (Avg Size: {G.number_of_nodes() / len(counts):.1f})")

    print(f"\nSaving updated graph to {GRAPH_FILE}...")
    nx.write_gexf(G, GRAPH_FILE)
    print("Done.")

if __name__ == "__main__":
    detect_communities_standalone()
