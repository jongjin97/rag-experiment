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

# Vector Database Directory
# Persist ChromaDB in the project root to keep it accessible
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"

# Model Name
MODEL_NAME: str = "gpt-4.1-mini"

def init_directories() -> None:
    """Ensure critical directories exist."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    SAMSUNG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories on module import (optional, but convenient for scripts)
init_directories()
