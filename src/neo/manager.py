import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver

class Neo4jManager:
    """
    Manages connection and operations for Neo4j Graph Database.
    """
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver: Optional[Driver] = None
        self._verify_vars()

    def _verify_vars(self):
        if not self.uri:
            raise ValueError("Neo4j URI is required. Set NEO4J_URI environment variable or pass explicitly.")
        # Basic check, drivers usually handle auth errors on connect but good to see if vars are empty.
    
    def connect(self):
        """Establishes connection to the Neo4j database."""
        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                self.driver.verify_connectivity()
                print(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                print(f"Failed to connect to Neo4j: {e}")
                raise e

    def close(self):
        """Closes the connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def query(self, query: str, parameters: Dict[str, Any] = None, db: str = "experiment") -> List[Dict[str, Any]]:
        """
        Executes a Cypher query and returns the results as a list of dictionaries.
        """
        if not self.driver:
            self.connect()
        
        with self.driver.session(database=db) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def add_node(self, node_id: str, label: str, properties: Dict[str, Any] = None):
        """
        Adds or merges a node with the given label and properties.
        Uses MERGE to avoid duplicates based on 'id'.
        """
        properties = properties or {}
        # Ensure ID is in properties for safety in the queries (though we use node_id param)
        properties['id'] = node_id
        
        # Cypher: MERGE (n:Label {id: $id}) SET n += $props
        # Using dynamic labels in py-driver is a bit tricky, typically we interpolate label safely or stick to fixed schema.
        # For generic RAG, we might have dynamic types. 
        # Safe interpolation for label (assuming valid identifier):
        sanitized_label = "".join([c for c in label if c.isalnum() or c == "_"])
        if not sanitized_label:
            sanitized_label = "Entity"

        cypher = f"""
        MERGE (n:`{sanitized_label}` {{id: $id}})
        SET n += $props
        RETURN n
        """
        self.query(cypher, {"id": node_id, "props": properties})

    def add_edge(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None):
        """
        Adds or merges a relationship between two nodes.
        Nodes are matched by their 'id' property.
        """
        properties = properties or {}
        sanitized_rel = "".join([c for c in relation_type if c.isalnum() or c == "_"]).upper()
        if not sanitized_rel:
            sanitized_rel = "RELATED_TO"

        cypher = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:`{sanitized_rel}`]->(b)
        SET r += $props
        RETURN r
        """
        self.query(cypher, {"source_id": source_id, "target_id": target_id, "props": properties})

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves local context (neighbors) for a given node.
        """
        # Simple depth 1 query for now
        cypher = """
        MATCH (n {id: $id})-[r]-(m)
        RETURN n, r, m
        LIMIT 50
        """
        return self.query(cypher, {"id": node_id})

    def clear_database(self):
        """
        WARNING: Deletes everything in the database. Use with caution.
        """
        cypher = "MATCH (n) DETACH DELETE n"
        self.query(cypher)
        print("Neo4j database cleared.")
