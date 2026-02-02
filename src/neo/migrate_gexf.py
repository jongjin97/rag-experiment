import os
import sys
import networkx as nx
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.neo.manager import Neo4jManager
from src.config import DATA_DIR

def migrate_gexf_to_neo4j(gexf_path: str):
    """
    Reads a GEXF file and imports it into Neo4j.
    """
    if not os.path.exists(gexf_path):
        print(f"Error: File not found at {gexf_path}")
        return

    print(f"Reading GEXF file: {gexf_path}...")
    try:
        graph = nx.read_gexf(gexf_path)
    except Exception as e:
        print(f"Failed to read GEXF: {e}")
        return

    print(f"Graph loaded: {graph.number_of_nodes()} Nodes, {graph.number_of_edges()} Edges.")

    # Connect to Neo4j
    manager = Neo4jManager()
    try:
        manager.connect()
    except Exception as e:
        print(f"Neo4j Connection Failed: {e}")
        return

    # Optional: Clear DB
    # print("Clearing existing database...")
    # manager.clear_database()

    # Import Nodes
    print("Importing Nodes...")
    # Batch processing could be added to manager, but for now we loop.
    # To speed up, we could optimize Neo4jManager to accept lists, 
    # but let's see performance first. 11MB file is ~10-50k nodes max.
    
    for node_id, data in tqdm(graph.nodes(data=True), desc="Nodes"):
        # GEXF attributes are in 'data' dict
        # 'label' might be the node ID or separate.
        # Ensure we have a type.
        node_type = data.get('type', 'Entity')
        # Rename 'type' to avoid collision if needed, but 'type' is good for label if it's broad.
        # But in Neo4j, label is the main category.
        
        # Clean data for Neo4j (remove complex objects if any, GEXF usually simple types)
        clean_props = {k: v for k, v in data.items() if k != 'type'}
        
        # We use the node_id from GEXF as the 'id' property in Neo4j
        manager.add_node(node_id, node_type, clean_props)

    # Import Edges
    print("Importing Edges...")
    for u, v, data in tqdm(graph.edges(data=True), desc="Edges"):
        # GEXF edge attributes
        rel_type = data.get('label') or data.get('relation') or "r"
        # If 'id' is in edge data, we can ignore it or store as property
        clean_props = {k: v for k, v in data.items() if k not in ['id', 'label', 'relation']}
        
        manager.add_edge(u, v, rel_type, clean_props)

    print("Migration Complete.")
    manager.close()

if __name__ == "__main__":
    load_dotenv()
    # Path from user request: data/light_rag/indexer/lightrag_graph.gexf
    # But let's verify if that's the one or the one in parent. 
    # The user asked specifically for "data/light_rag/indexer/lightrag_graph.gexf".
    # I will construct that path.
    
    target_file = DATA_DIR / "processed_experiment" / "batch_input" / "graph.gexf"
    
    # Fallback to parent if not found (just in case user was slightly off, 
    # but user was specific so we try specific first)
    if not target_file.exists():
        print(f"File not found at {target_file}, checking parent dir...")
        target_file = DATA_DIR / "light_rag" / "lightrag_graph.gexf"

    migrate_gexf_to_neo4j(str(target_file))
