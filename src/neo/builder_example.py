import os
import sys
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.neo.manager import Neo4jManager

def main():
    load_dotenv()
    
    # Initialize Manager
    manager = Neo4jManager()
    
    print("Connecting to Neo4j...")
    try:
        manager.connect()
    except Exception as e:
        print(f"Could not connect, are env vars set? {e}")
        return

    # Clear DB for fresh start (Optional, commented out)
    # manager.clear_database()

    print("Building Example Graph...")
    
    # Add Nodes
    entities = [
        {"id": "Samsung_Electronics", "type": "Company", "desc": "A major Korean tech company."},
        {"id": "Seoul", "type": "Location", "desc": "Capital of South Korea."},
        {"id": "Galaxy_S24", "type": "Product", "desc": "Latest smartphone model."},
        {"id": "AI_Chip", "type": "Technology", "desc": "Semiconductor for AI processing."}
    ]

    for e in entities:
        print(f"Adding Node: {e['id']}")
        manager.add_node(e['id'], e['type'], {"description": e['desc']})

    # Add Edges
    relationships = [
        ("Samsung_Electronics", "Seoul", "HEADQUARTERED_IN", {"since": "1969"}),
        ("Samsung_Electronics", "Galaxy_S24", "MANUFACTURES", {}),
        ("Galaxy_S24", "AI_Chip", "CONTAINS", {"model": "Exynos 2400"})
    ]

    for src, tgt, rel, props in relationships:
        print(f"Adding Edge: {src} -[{rel}]-> {tgt}")
        manager.add_edge(src, tgt, rel, props)

    print("Graph Build Complete.")
    manager.close()

if __name__ == "__main__":
    main()
