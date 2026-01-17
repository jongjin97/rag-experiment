from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import HUGGINGFACE_EMBEDDING_MODEL
# Using the same model as Naive RAG for consistency

def get_embedding_function():
    """Returns the HuggingFace embedding function instance."""
    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
