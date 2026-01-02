import os
from typing import List, Optional
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHROMA_DB_DIR, DATA_DIR, MODEL_NAME
from src.utils.document_loader import load_documents

# Configuration matches chunking.py v2 settings
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DOCUMENT_DIR = DATA_DIR / "samsung"

def get_embeddings():
    """Initialize and return the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore(chunk_size: int = 256):
    """
    Get the existing ChromaDB vector store.
    Assumes the collection has been created by chunking.py
    """
    embedding_function = get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_function,
        collection_name=f"samsung_chunking_{chunk_size}_v2",
    )

def split_documents(documents, chunk_size=256, chunk_overlap=20):
    """
    Split documents for BM25 indexing. 
    MUST match the splitting logic used in ingestion to ensure alignment.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def get_bm25_retriever(chunk_size: int = 256):
    """
    Builds a BM25 Retriever by fetching documents from the existing ChromaDB.
    """
    print(f"Fetching documents from ChromaDB for chunk size {chunk_size}...")
    vectorstore = get_vectorstore(chunk_size)
    
    # Fetch all documents from the collection
    result = vectorstore.get() 
    
    texts = result['documents']
    metadatas = result['metadatas']
    
    if not texts:
         raise ValueError(f"No documents found in ChromaDB collection for chunk size {chunk_size}")

    docs = []
    for text, metadata in zip(texts, metadatas):
        docs.append(Document(page_content=text, metadata=metadata))
        
    print(f"Loaded {len(docs)} documents from ChromaDB.")
    
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 5
    return retriever

def get_ensemble_retriever(chunk_size: int = 256, alpha: float = 0.5):
    """
    Returns a Hybrid Retriever (Vector + BM25).
    Alpha: Weight for Sparse (BM25). 
           However, EnsembleRetriever uses 'weights' list [bm25_weight, dense_weight].
           If alpha is for BM25, then:
           weights = [alpha, 1 - alpha]
    """
    bm25 = get_bm25_retriever(chunk_size)
    
    vectorstore = get_vectorstore(chunk_size)
    dense = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print(f"Initializing Ensemble Retriever with BM25 weight={alpha}, Dense weight={1-alpha}")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[alpha, 1.0 - alpha]
    )
    return ensemble_retriever

def get_hyde_retriever(chunk_size: int = 256):
    """
    Returns a HyDE (Hypothetical Document Embeddings) Retriever.
    Generates a hypothetical answer, embeds it, and searches.
    """
    vectorstore = get_vectorstore(chunk_size)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    # HyDE Prompt: Generate a scientific/financial passage to answer the question
    template = """Please write a passage to answer the question
Question: {question}
Passage:"""
    prompt = PromptTemplate.from_template(template)
    
    # Custom HyDE Chain: Query -> LLM -> Hypothetical Doc -> Retrieval
    def hyde_search(query_str):
        hypothetical_doc = (prompt | llm | StrOutputParser()).invoke({"question": query_str})
        print(f"\n[HyDE] Generated Hypothetical Doc: {hypothetical_doc[:100]}...")
        return base_retriever.invoke(hypothetical_doc)
        
    return hyde_search

if __name__ == "__main__":
    # Test block
    print("Testing Retrievers...")
    
    # Test Ensemble
    hybrid = get_ensemble_retriever(chunk_size=256, alpha=0.5)
    docs = hybrid.invoke("DX 부문의 매출은 얼마인가?")
    print(f"Hybrid Retrieved {len(docs)} docs")
    print(docs[0].page_content[:100])
    
    # Test HyDE
    # hyde = get_hyde_retriever(chunk_size=256)
    # docs = hyde("DX 부문의 매출은 얼마인가?")
    # print(f"HyDE Retrieved {len(docs)} docs")