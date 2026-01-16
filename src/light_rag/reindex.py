import asyncio
import shutil
import sys
import os
from pathlib import Path

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.light_rag.lightrag import LightRAG
from src.config import CHROMA_DB_DIR

async def main():
    print("Re-indexing LightRAG System...")
    
    # 1. Clear existing Vector DB
    if CHROMA_DB_DIR.exists():
        print(f"Removing existing ChromaDB at {CHROMA_DB_DIR}...")
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            print("Successfully removed old database.")
        except Exception as e:
            print(f"Error removing database: {e}")
            return
    else:
        print("No existing database found.")

    # 2. Initialize LightRAG (this will create new empty collections)
    print("Initializing LightRAG...")
    try:
        rag = LightRAG()
    except Exception as e:
        print(f"Error initializing LightRAG: {e}")
        return

    # 3. Trigger Indexing
    print("Starting Indexing Process (this may take a while)...")
    try:
        await rag.index()
        print("Indexing Successfully Completed!")
    except Exception as e:
        print(f"Error during indexing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
