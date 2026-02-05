import chromadb
from typing import List, Dict, Any
from src.light_rag_v2.storage.base import BaseVectorStorage
from src.config import CHROMA_DB_DIR

class VectorStorage(BaseVectorStorage):
    """
    Vector Storage implementation using ChromaDB native client.
    """
    def __init__(self, collection_name: str):
        # Initialize PersistentClient to save data to disk
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        
        # Get or create the collection
        # We don't define an embedding function here because we pass embeddings directly
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    async def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        formatted_results = []
        for i, id in enumerate(ids):
            item = {
                "id": id,
                "metadata": metadatas[i] if metadatas else {},
                "score": 1 - distances[i] if distances else 0 # Convert distance to similarity score if needed
            }
            formatted_results.append(item)
            
        return formatted_results

    async def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        self.collection.upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[metadata]
        )
