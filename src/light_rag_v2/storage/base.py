from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union
import numpy as np

class BaseVectorStorage(ABC):
    """Abstract base class for Vector Storage."""
    
    @abstractmethod
    async def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve top_k similar items."""
        pass

    @abstractmethod
    async def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Insert or update a vector."""
        pass

class BaseGraphStorage(ABC):
    """Abstract base class for Graph Storage."""
    
    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data."""
        pass

    @abstractmethod
    async def get_node_degree(self, node_id: str) -> int:
        """Get degree of a node."""
        pass

    @abstractmethod
    async def get_edges(self, node_id: str) -> List[tuple[str, str, Dict[str, Any]]]:
        """Get all edges connected to a node (source, target, data)."""
        pass

class BaseKVStorage(ABC):
    """Abstract base class for Key-Value Storage."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        pass
