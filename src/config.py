import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Root Directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMSUNG_DIR: Path = DATA_DIR / "samsung"

# GraphRAG
GRAPH_RAG_DIR: Path = DATA_DIR / "graph_rag"
KNOWLEDGE_GRAPH_PATH: Path = GRAPH_RAG_DIR / "knowledge_graph.gexf"

# GraphRAG v2
GRAPH_RAG_V2_DIR: Path = DATA_DIR / "graph_rag_v2"
KNOWLEDGE_GRAPH_V2_PATH: Path = GRAPH_RAG_V2_DIR / "knowledge_graph.gexf"
LIGHT_RAG_DIR: Path = DATA_DIR / "light_rag" # Artifacts for LightRAG
LIGHT_RAG_V2_DIR: Path = DATA_DIR / "light_rag_v2" # Artifacts for LightRAG v2
KNOWLEDGE_LIGTH_GRAPH_PATH: Path = LIGHT_RAG_DIR / "lightrag_graph.gexf" # Updated to new graph
KNOWLEDGE_LIGTH_GRAPH_V2_PATH: Path = LIGHT_RAG_V2_DIR / "lightrag_graph.gexf" # Updated to new graph


SUBMIT_BATCH_SIZE: int = 50 # submitting to LLM
MAP_BATCH_SIZE: int = 10     # parallel map processing
MAX_COMMUNITY_TOKENS: int = 2000 # max tokens for a single community summary

# Vector Database Directory
# Persist ChromaDB in the project root to keep it accessible
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"

# Model Name
MODEL_NAME: str = "gpt-4o-mini"
GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
DEEPSEEK_MODEL_NAME: str = "deepseek-reasoner"
HUGGINGFACE_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def init_directories() -> None:
    """Ensure critical directories exist."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    SAMSUNG_DIR.mkdir(parents=True, exist_ok=True)
    LIGHT_RAG_V2_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories on module import (optional, but convenient for scripts)
init_directories()
