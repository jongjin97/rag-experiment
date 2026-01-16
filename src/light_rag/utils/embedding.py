from langchain_community.embeddings import HuggingFaceEmbeddings

# Using the same model as Naive RAG for consistency
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_function():
    """Returns the HuggingFace embedding function instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
