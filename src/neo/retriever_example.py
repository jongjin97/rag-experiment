import os
import sys
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.neo.manager import Neo4jManager

def main():
    load_dotenv()
    
    manager = Neo4jManager()
    try:
        manager.connect()
    except:
        print("Connection failed.")
        return

    target_node = "Samsung_Electronics"
    print(f"Retrieving Neighbors for: {target_node}")
    
    neighbors = manager.get_neighbors(target_node)
    
    if not neighbors:
        print("No neighbors found or node does not exist.")
    else:
        for record in neighbors:
            # Cypher return: n, r, m
            n = record.get('n')
            r = record.get('r')
            m = record.get('m')
            
            # Neo4j driver returns Node/Relationship objects which are dict-like
            # but getting properties might look like this:
            other_node_props = m if isinstance(m, dict) else dict(m)
            rel_type = r[1] if isinstance(r, tuple) else "REL" # Should inspect 'r' object properly
            
            # Usually record['r'] is a relationship object that has .type
            # Let's just print the raw record to see structure if unsure,
            # but clean output is better.
            
            # Ideally manager returns pure dicts via .data(), so:
            print(f" - Connected to: {other_node_props.get('id', 'Unknown')} (Props: {other_node_props})")

    manager.close()

if __name__ == "__main__":
    main()
