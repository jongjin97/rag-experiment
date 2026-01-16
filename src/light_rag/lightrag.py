import asyncio
from typing import Optional
from src.light_rag.storage.graph import GraphStorage
from src.light_rag.storage.vector import VectorStorage
from src.light_rag.retriever import LightRAGRetriever
from src.light_rag.utils.gexf_loader import load_gexf_to_storage
from src.light_rag.prompt import PROMPT_ANSWER_GENERATION
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.config import OPENAI_API_KEY, MODEL_NAME

class LightRAG:
    """
    Main class for LightRAG System.
    """
    def __init__(self):
        # 1. Initialize Storages
        self.graph_storage = GraphStorage()
        self.entity_vector_storage = VectorStorage(collection_name="lightrag_entities")
        self.relation_vector_storage = VectorStorage(collection_name="lightrag_relations")
        self.chunk_vector_storage = VectorStorage(collection_name="lightrag_chunks")
        
        # 2. Initialize Retriever
        self.retriever = LightRAGRetriever(
            entity_vector_storage=self.entity_vector_storage,
            relation_vector_storage=self.relation_vector_storage,
            chunk_vector_storage=self.chunk_vector_storage,
            graph_storage=self.graph_storage
        )
        
        # 3. Initialize Generator (LLM)
        # Using OpenAI for generation as per config, or fallback
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME, 
            temperature=0,
            api_key=OPENAI_API_KEY
        )

    async def index(self):
        """
        Builds the index by loading the GEXF file into vector stores.
        """
        print("Starting LightRAG Indexing Process...")
        await load_gexf_to_storage(
            graph_storage=self.graph_storage,
            entity_vector_storage=self.entity_vector_storage,
            relation_vector_storage=self.relation_vector_storage
        )
        print("Indexing Complete.")

    async def query(self, query: str, mode: str = "hybrid", top_k: int = 5) -> str:
        """
        Answers a query using the specified retrieval mode.
        Modes: 'low', 'high', 'hybrid'
        """
        # 1. Retrieve Context
        context = ""
        if mode == "low":
            context = await self.retriever.retrieve_low_level(query, top_k=top_k)
        elif mode == "high":
            context = await self.retriever.retrieve_high_level(query, top_k=top_k)
        else:
            context = await self.retriever.retrieve_hybrid(query, top_k=top_k)
            
        print("Context:", context)
        # 2. Generate Answer
        prompt = PromptTemplate(
            template=PROMPT_ANSWER_GENERATION,
            input_variables=["context_str", "query_str"]
        )
        chain = prompt | self.llm
        response = await chain.ainvoke({"context_str": context, "query_str": query})
        
        return response.content

# Singleton instance helper if needed
_lightrag_instance = None

def get_lightrag_system():
    global _lightrag_instance
    if _lightrag_instance is None:
        _lightrag_instance = LightRAG()
    return _lightrag_instance
