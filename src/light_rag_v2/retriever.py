import json
from pathlib import Path
from typing import List, Dict, Any, Set
from src.light_rag_v2.storage.graph import GraphStorage
from src.light_rag_v2.storage.vector import VectorStorage
from src.light_rag_v2.utils.embedding import get_embedding_function
from src.config import LIGHT_RAG_V2_DIR

class LightRAGRetriever:
    """
    Handles Multi-Level Retrieval for LightRAG.
    Levels:
    1. Low-Level: Entity + neighbors
    2. High-Level: Relationships
    3. Chunk-Level: Original text chunks (via Vector Search)
    """
    def __init__(
        self,
        entity_vector_storage: VectorStorage,
        relation_vector_storage: VectorStorage,
        chunk_vector_storage: VectorStorage,
        graph_storage: GraphStorage
    ):
        self.entity_vector_storage = entity_vector_storage
        self.relation_vector_storage = relation_vector_storage
        self.chunk_vector_storage = chunk_vector_storage
        self.graph_storage = graph_storage
        self.embedding_fn = get_embedding_function()
        
        # Load chunks map for content lookup
        self.chunks_map = {}
        chunks_file = LIGHT_RAG_V2_DIR / "indexer" / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks_map = json.load(f)

    async def retrieve_chunks(self, query: str, top_k: int = 5) -> str:
        """
        Retrieves original text chunks based on query similarity.
        Lookups content from chunks.json using the retrieved ID.
        """
        query_vec = self.embedding_fn.embed_query(query)
        results = await self.chunk_vector_storage.query(query_vec, top_k=top_k)
        
        context_parts = []
        seen_content = set()
        for res in results:
            chunk_id = res['id']

            print(f"[DEBUG] Chunk: {chunk_id}, Score: {res.get('score'):.4f}")
            # Lookup content
            chunk_data = self.chunks_map.get(chunk_id, {})
            content = chunk_data.get("content", "")
            source = chunk_data.get("source", "unknown")
            if content in seen_content:
                continue
            seen_content.add(content)
            if content:
                text = f"{content}"
                context_parts.append(text)
        print("context_parts:", context_parts)
        print("**"*100)
        return "\n\n".join(context_parts)

    async def retrieve_low_level(self, query: str, top_k: int = 5) -> str:
        """
        Low-Level Retrieval: Focuses on specific entities.
        """
        query_vec = self.embedding_fn.embed_query(query)
        
        # Search for top entities
        results = await self.entity_vector_storage.query(query_vec, top_k=top_k)
        
        context_parts = []
        visited_nodes: Set[str] = set()

        for res in results:
            node_id = res['id']
            print(f"[DEBUG] Entity: {node_id}, Score: {res.get('score'):.4f}")
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)
            
            # Add Entity Info
            metadata = res['metadata']
            entity_desc = f"Entity: {metadata.get('entity_name')} ({metadata.get('type')})\nDescription: {metadata.get('description')}"
            context_parts.append(entity_desc)
            
            # Add 1-hop Neighbors (Relations)
            edges = await self.graph_storage.get_edges(node_id)
            for _, target, data in edges[:5]:
                rel_desc = f"  - related to {target} via {data.get('relation')}: {data.get('description')}"
                context_parts.append(rel_desc)
                
        return "\n".join(context_parts)

    async def retrieve_high_level(self, query: str, top_k: int = 5) -> str:
        """
        High-Level Retrieval: Focuses on broader relationships.
        """
        query_vec = self.embedding_fn.embed_query(query)
        results = await self.relation_vector_storage.query(query_vec, top_k=top_k)
        
        context_parts = []
        for res in results:
            metadata = res['metadata']
            print(f"[DEBUG] Rel High: {metadata.get('source')} -> {metadata.get('target')}, Score: {res.get('score'):.4f}")
            rel_text = (
                f"Relationship: {metadata.get('source')} -> {metadata.get('target')}\n"
                f"Type: {metadata.get('relation')}\n"
                f"Context: {metadata.get('description')}"
            )
            context_parts.append(rel_text)
            
        return "\n".join(context_parts)

    async def retrieve_hybrid(self, query: str, top_k: int = 5) -> str:
        """
        Combines Low, High, and Chunk context.
        """
        chunk_context = await self.retrieve_chunks(query, top_k=top_k)
        low_level_context = await self.retrieve_low_level(query, top_k=top_k)
        high_level_context = await self.retrieve_high_level(query, top_k=top_k)
        
        return (
            f"=== Original Text Context ===\n{chunk_context}\n\n"
            f"=== Entity Context ===\n{low_level_context}\n\n"
            f"=== Relationship Context ===\n{high_level_context}"
        )
