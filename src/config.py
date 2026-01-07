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
SUBMIT_BATCH_SIZE: int = 50 # submitting to LLM
MAP_BATCH_SIZE: int = 10     # parallel map processing
MAX_COMMUNITY_TOKENS: int = 2000 # max tokens for a single community summary

# Vector Database Directory
# Persist ChromaDB in the project root to keep it accessible
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"

# Model Name
MODEL_NAME: str = "gpt-4.1-nano"
GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
DEEPSEEK_MODEL_NAME: str = "deepseek-reasoner"

def init_directories() -> None:
    """Ensure critical directories exist."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    SAMSUNG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories on module import (optional, but convenient for scripts)
init_directories()
