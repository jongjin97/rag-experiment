from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import CHROMA_DB_DIR, DATA_DIR
from src.utils.document_loader import load_documents

# Embeddings Model Configuration
# Using a lightweight, high-performance model suitable for CPU/local usage
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """Initialize and return the embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore():
    """Initialize and return the ChromaDB vector store."""
    embedding_function = get_embeddings()
    # persistent_client=None allows Chroma to manage the connection automatically via persist_directory
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_function,
        collection_name="samsung_report_rag"
    )

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for flexible retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def ingest_data():
    """
    Main function to load texts, split them, and index them into ChromaDB.
    """
    # 1. Load Documents
    print("Loading documents...")
    # Loading both papers and samsung reports if needed. Currently focusing on Samsung.
    # You can change this to DATA_DIR to load everything recursively if load_documents supports it,
    # or specify the subfolder.
    target_dir = DATA_DIR / "samsung"
    raw_documents = load_documents(target_dir)
    
    if not raw_documents:
        print("No documents found to ingest.")
        return

    print(f"Loaded {len(raw_documents)} documents.")

    # 2. Split Text (Chunking)
    print("Splitting documents...")
    chunks = split_documents(raw_documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Index to ChromaDB
    print("Indexing to ChromaDB (this may take a while)...")
    vectorstore = get_vectorstore()
    
    # Add documents in batches to avoid hitting limits (optional, but good practice)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    print("Ingestion complete!")

if __name__ == "__main__":
    vectorstore = get_vectorstore()
    print(vectorstore.get(limit=5))
