from src.naive_rag.retriever import get_retriever
from src.config import CHROMA_DB_DIR
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def debug_retrieval():
    print("--- 1. Checking if the correct information exists in DB ---")
    
    # Connect directly to Chroma to inspect data
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=str(CHROMA_DB_DIR), collection_name="samsung_report_rag", embedding_function=embedding_function)
    
    # Brute-force search for the expected content (we know it contains "DX 부문" and "TV" from previous loader logs)
    # Note: This is efficient enough for small-medium datasets (~2000 chunks)
    all_data = db.get() # Get all ids and documents
    found_count = 0
    target_keywords = ["DX 부문", "TV"]
    
    print(f"Total chunks in DB: {len(all_data['ids'])}")
    
    for i, doc_text in enumerate(all_data['documents']):
        if all(k in doc_text for k in target_keywords):
            print(f"\n[FOUND MATCHING CHUNK ID: {all_data['ids'][i]}]")
            print(f"Content Preview:\n{doc_text[:300]}...") # Print first 300 chars
            found_count += 1
            if found_count >= 3: break
            
    if found_count == 0:
        print("\n[CRITICAL] Could not find any chunk containing both 'DX 부문' and 'TV'.")
        print("Possible causes: Chunking split the table header from body, or Table extraction failed.")
        
    print("\n\n--- 2. Analyzing Retrieval Performance ---")
    query = "DX 부문의 주요 제품은 무엇인가?"
    print(f"Query: {query}")
    
    # Use similarity_search_with_score to see distances
    # Lower score is better for L2 distance (or cosine distance depending on Chroma setup)
    results = db.similarity_search_with_score(query, k=5)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n[Rank {i+1}] Score: {score:.4f}")
        print(f"Content Preview: {doc.page_content.replace(chr(10), ' ')[:100]}...")

if __name__ == "__main__":
    debug_retrieval()
