from src.utils.document_loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR, MODEL_NAME, DATA_DIR
from typing import List
from langchain_core.documents import Document
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance, Faithfulness
import os
import json
import asyncio
import numpy as np

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DOCUMENT_DIR = DATA_DIR / "samsung"

def get_embeddings():
    """Initialize and return the embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore(chunk_size):
    """Initialize and return the ChromaDB vector store."""
    embedding_function = get_embeddings()
    # persistent_client=None allows Chroma to manage the connection automatically via persist_directory
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_function,
        collection_name=f"samsung_chunking_{chunk_size}_v2",
    )

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents into smaller chunks for flexible retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


async def context_relevance_metrics(user_input, retrieved_contexts):
    scorer = ContextRelevance(
        llm=llm_factory(MODEL_NAME, client=AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))),
    )

    result = await scorer.ascore(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
    )
    return result

async def main():
    eval_dataset = json.load(open(DATA_DIR / "eval_dataset.json", "r", encoding="utf-8"))
    original_documents = load_documents(DOCUMENT_DIR)
    if not original_documents:
        print("No documents found!")
        return

    chunk_sizes = [128, 256, 512, 1024]
    chunk_overlap = 20
    
    all_results = []
    
    for chunk_size in chunk_sizes:
        print(f"\n=== Processing Chunk Size: {chunk_size} ===")
        vectorstore = get_vectorstore(chunk_size)

        if vectorstore._collection.count() > 0:
            print(f"Vectorstore already exists for chunk size {chunk_size}. Skipping...")
        else:
            # Split documents and add to vectorstore
            documents = split_documents(original_documents, chunk_size, chunk_overlap)
                
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                print(f"Indexed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        print("\nEvaluating queries...")

        chunk_results = []

        for dataset in eval_dataset:
            retrieved_docs = retriever.invoke(dataset["user_input"])
            retrieved_docs = [doc.page_content for doc in retrieved_docs]
            result = await context_relevance_metrics(dataset["user_input"], retrieved_docs)
            chunk_results.append(result.value)

        all_results.append({
            "chunk_size": chunk_size,
            "eval_values": chunk_results,
            "mean": np.mean(chunk_results),
            "std": np.std(chunk_results)
        })

    with open(DATA_DIR / "chunking_results_v2.txt", "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(f"Chunk Size: {result['chunk_size']}, Mean: {result['mean']}, Std: {result['std']}\n")

    # Save results
    with open(DATA_DIR / "chunking_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\nExperiment complete. Results saved to chunking_results_v2.json")

if __name__ == "__main__":
    asyncio.run(main())



    

    

    