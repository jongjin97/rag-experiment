from typing import Any, List, Dict, Optional
import networkx as nx
from src.light_rag_v2.storage.base import BaseGraphStorage
from src.config import KNOWLEDGE_LIGTH_GRAPH_V2_PATH

class GraphStorage(BaseGraphStorage):
    def __init__(self):
        self.graph = nx.Graph()
        if KNOWLEDGE_LIGTH_GRAPH_V2_PATH.exists():
            print(f"Loading Knowledge Graph from {KNOWLEDGE_LIGTH_GRAPH_V2_PATH}...")
            self.graph = nx.read_gexf(str(KNOWLEDGE_LIGTH_GRAPH_V2_PATH))
            print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        else:
            print(f"Warning: Graph file not found at {KNOWLEDGE_LIGTH_GRAPH_V2_PATH}. Initializing empty graph.")

    async def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None
    
    async def get_node_degree(self, node_id: str) -> int:
        if self.graph.has_node(node_id):
            return self.graph.degree[node_id] # type: ignore
        return 0
    
    async def get_edges(self, node_id: str) -> List[tuple[str, str, Dict[str, Any]]]:
        if self.graph.has_node(node_id):
            # networkx edges(data=True) returns (u, v, data)
            return list(self.graph.edges(node_id, data=True))
        return []
