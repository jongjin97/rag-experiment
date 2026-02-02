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

GRAPH_FILE = DATA_DIR / "processed_experiment" / "batch_input2" /"graph.gexf"

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
        {"level": 3, "max_comm_size": 10, "resolution": 0.1}, # Bottom
        {"level": 2, "max_comm_size": 20, "resolution": 0.2}, 
        {"level": 1, "max_comm_size": 100, "resolution": 0.3},
        {"level": 0, "max_comm_size": None, "resolution": 0.4} # Top
    ]
    
    # --- Super Node Removal Strategy ---
    # Temporarily remove "삼성전자" to allow sub-hubs (DS, DX, etc.) to form distinct clusters.
    SUPER_NODE = "삼성전자"
    removed_super_node = False
    
    if G.has_node(SUPER_NODE):
        print(f"\n[Strategy] Temporarily removing Super Node '{SUPER_NODE}' for detection...")
        current_graph = G.copy()
        current_graph.remove_node(SUPER_NODE)
        removed_super_node = True
    else:
        print(f"\n[Strategy] Super Node '{SUPER_NODE}' not found. Proceeding normally.")
        current_graph = G

    # Mapping from current_graph node -> original entity name (list)
    # Ideally we track: original ID -> L3 ID -> L2 ID -> L1 ID -> L0 ID
    
    # Initialize hierarchy map: node_name -> {level: comm_id}
    # Note: hierarchy initially only covers nodes in current_graph (excluding Samsung)
    hierarchy = {n: {} for n in current_graph.nodes()}
    
    # current_node_to_members maps the ID in 'current_graph' to list of original entity names
    current_node_to_members = {n: [n] for n in current_graph.nodes()}
    
    prev_partition_count = 0 
    
    for i, config in enumerate(levels_config):
        level = config["level"]
        max_size = config["max_comm_size"] # Cap for THIS level
        resolution = config.get("resolution", 1.0)
        
        print(f"\n--- Running Level {level} Detection ---")
        print(f"  - Input Graph: {current_graph.number_of_nodes()} nodes")
        
        # 1. Convert to iGraph & Run Leiden
        node_names = list(current_graph.nodes())
        node_map = {name: i for i, name in enumerate(node_names)}
        
        edges = []
        weights = []
        for u, v, data in current_graph.edges(data=True):
            if u in node_map and v in node_map:
                edges.append((node_map[u], node_map[v]))
                weights.append(data.get('weight', 1.0))
        
        ig_graph = ig.Graph(n=len(node_names), edges=edges, directed=False)
        if weights: ig_graph.es['weight'] = weights
        
        # Leiden with Resolution
        kwargs = {
            'weights': 'weight' if weights else None,
            'n_iterations': 5,
            'seed': 42,
            'resolution_parameter': resolution
        }
        if max_size: kwargs['max_comm_size'] = max_size
            
        partition = leidenalg.find_partition(ig_graph, leidenalg.CPMVertexPartition, **kwargs)
            
        print(f"  -> Detected {len(partition)} communities.")
        
        # 2. Assign Attributes & Identify Candidates for Next Level
        merge_threshold = 5 if max_size is None or max_size > 5 else 2 
        
        candidates = [] 
        next_node_to_members = defaultdict(list)
        current_level_mapping = {} # current_node_name -> new_comm_id
        
        for c_id, cluster_nodes in enumerate(partition):
            size = len(cluster_nodes)
            
            # Constraint Check
            is_candidate = True
            if max_size:
                is_candidate = (size >= int(max_size * 0.9)) # 90% full
                
            unique_c_id = f"L{level}_{c_id}" 
            
            if is_candidate:
                candidates.append(unique_c_id)
                
            for node_idx in cluster_nodes:
                current_node_name = node_names[node_idx]
                current_level_mapping[current_node_name] = unique_c_id
                
                original_members = current_node_to_members[current_node_name]
                for member in original_members:
                    hierarchy[member][level] = unique_c_id
                
                if is_candidate:
                    next_node_to_members[unique_c_id].extend(original_members)

        print(f"  -> {len(candidates)} communities hit constraint and will merge.")
        
        if level == 0 or len(candidates) < 2:
            print("  -> Stopping recursion (Top level or no candidates).")
            break

        # 3. Build Induced Graph
        print("  -> Building Induced Graph...")
        G_next = nx.Graph()
        G_next.add_nodes_from(candidates)
        
        edge_log = defaultdict(float)
        
        for u, v, data in current_graph.edges(data=True):
            c1 = current_level_mapping.get(u)
            c2 = current_level_mapping.get(v)
            
            if c1 in candidates and c2 in candidates and c1 != c2:
                key = tuple(sorted((c1, c2)))
                edge_log[key] += data.get('weight', 1.0)
                
        for (c1, c2), w in edge_log.items():
            G_next.add_edge(c1, c2, weight=w)
            
        current_graph = G_next
        current_node_to_members = next_node_to_members

    # --- Finalize Hierarchy & Super Node Restoration ---
    print("\nApplying Hierarchy to NetworkX Graph...")
    
    # 1. Fill gaps for existing nodes
    for node, levels in hierarchy.items():
        last_val = levels.get(3)
        for lvl in [2, 1, 0]:
             if lvl not in levels and last_val is not None:
                 levels[lvl] = last_val
             elif lvl in levels:
                 last_val = levels[lvl]
                 
    # 2. Restore Super Node using Majority Voting
    if removed_super_node and G.has_node(SUPER_NODE):
        print(f"[Restoration] Assigning community to '{SUPER_NODE}' via Majority Voting...")
        
        # Get neighbors
        neighbors = list(G.neighbors(SUPER_NODE))
        
        # For each level, find dominant community
        super_node_comms = {}
        from collections import Counter
        
        for lvl in [3, 2, 1, 0]:
            neighbor_comms = []
            for nbr in neighbors:
                if nbr in hierarchy and lvl in hierarchy[nbr]:
                    neighbor_comms.append(hierarchy[nbr][lvl])
            
            if neighbor_comms:
                # Pick most common
                most_common = Counter(neighbor_comms).most_common(1)[0][0]
                super_node_comms[lvl] = most_common
            else:
                # Fallback if disconnected
                super_node_comms[lvl] = "L0_0" # Default or create new
                
        hierarchy[SUPER_NODE] = super_node_comms
        print(f"  -> Assigned to: {super_node_comms}")

    for node_name, levels in hierarchy.items():
        for level, c_id in levels.items():
            G.nodes[node_name][f'community_level_{level}'] = c_id
            if level == 3:
                G.nodes[node_name]['community'] = c_id

    print("\nStats:")
    for level in [3, 2, 1, 0]:
        counts = defaultdict(int)
        for n, d in G.nodes(data=True):
            cid = d.get(f'community_level_{level}')
            if cid is not None:
                counts[cid] += 1
        
        num_comms = len(counts)
        avg_size = G.number_of_nodes() / num_comms if num_comms > 0 else 0
        sorted_sizes = sorted(counts.values(), reverse=True)
        
        print(f"  Level {level} Communities: {num_comms} (Avg Size: {avg_size:.1f})")
        
        if num_comms <= 200:
            print(f"    - Sizes: {sorted_sizes}")
        else:
            print(f"    - Top 20 Sizes: {sorted_sizes[:20]} ...")
            print(f"    - Smallest 5 Sizes: {sorted_sizes[-5:]}")

    print(f"\nSaving updated graph to {GRAPH_FILE}...")
    nx.write_gexf(G, GRAPH_FILE)
    print("Done.")

if __name__ == "__main__":
    detect_communities_standalone()
