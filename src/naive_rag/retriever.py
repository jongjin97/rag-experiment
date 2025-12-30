from src.config import CHROMA_DB_DIR
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_retriever(k: int = 5):
    # 1. Initialize Vector Store (Dense Retriever)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name="samsung_report_rag",
        embedding_function=embedding_function
    )
    dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # 2. Initialize BM25 Retriever (Sparse Retriever)
    # We need to load all documents to initialize BM25. 
    # NOTE: In production with huge datasets, you'd load a pre-saved BM25 index. 
    # For <5000 chunks, loading from Chroma is acceptable.
    all_docs = vectorstore.get()['documents']
    metadatas = vectorstore.get()['metadatas']
    
    # Reconstruct Document objects for BM25
    docs_for_bm25 = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_docs, metadatas)
    ]
    
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = k

    # 3. Create Ensemble Retriever
    # Weight: 0.5 for BM25 (Keyword), 0.5 for Dense (Semantic)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.6, 0.4] # Give slightly more weight to keywords for specific entities like "DX 부문"
    )
    
    return ensemble_retriever

def retrieve_context(query: str, k: int = 5):
    retriever = get_retriever(k)
    return retriever.invoke(query)
