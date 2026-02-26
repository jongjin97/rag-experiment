import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List

from src.utils.document_loader import load_documents_merged
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.metrics.collections import ContextRelevance
from datasets import Dataset

from src.config import CHROMA_DB_DIR, MODEL_NAME, DATA_DIR

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
        collection_name=f"samsung_chunking_{chunk_size}_v3_baseline",
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
    # 1. 병합된 CSV 평가 데이터셋 로드
    eval_file_path = DATA_DIR / "eval_dataset_v3_merged.csv"
    if not eval_file_path.exists():
        print(f"평가 데이터셋을 찾을 수 없습니다: {eval_file_path}")
        return
    eval_df = pd.read_csv(eval_file_path, encoding="utf-8-sig")
    
    # 평가에 필요한 컬럼: user_input, reference 추출
    if "user_input" in eval_df.columns:
        questions = eval_df["user_input"].tolist()
    elif "question" in eval_df.columns:
        questions = eval_df["question"].tolist()
    else:
        print("CSV 파일에서 질문 컬럼(user_input 또는 question)을 찾을 수 없습니다.")
        return
        
    if "reference" in eval_df.columns:
        references = eval_df["reference"].tolist()
    elif "ground_truth" in eval_df.columns:
        references = eval_df["ground_truth"].tolist()
    else:
        print("CSV 파일에서 정답 컬럼(reference 또는 ground_truth)을 찾을 수 없습니다.")
        return

    # 전체 데이터셋을 사용하여 정확히 비교합니다.
    test_size = len(questions)
    print(f"총 {test_size}개의 전체 쿼리에 대해 평가를 진행합니다...")

    original_documents = load_documents_merged(DOCUMENT_DIR)
    if not original_documents:
        print("No documents found!")
        return

    chunk_sizes = [512, 1024]
    chunk_overlap = 20
    
    all_results = []
    
    for chunk_size in chunk_sizes:
        print(f"\n=== Processing Chunk Size: {chunk_size} ===")
        vectorstore = get_vectorstore(chunk_size)

        if vectorstore._collection.count() > 0:
            print(f"Vectorstore already exists for chunk size {chunk_size}. Skipping indexing...")
        else:
            # 테이블 복원 없이 순수 Text Splitter만 적용
            documents = split_documents(original_documents, chunk_size, chunk_overlap)
                
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                print(f"Indexed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        print("\nQuery 단위 Context 검색 및 평가 진행 중...")

        # Ragas v0.4.x 평가용 데이터 딕셔너리 구축
        eval_dict = {
            "user_input": questions,
            "retrieved_contexts": [],
            "reference": references
        }

        # Retrieval 및 수동 Context Relevance 수행
        context_relevance_scores = []
        for q in questions:
            retrieved_docs = retriever.invoke(q)
            docs_content = [doc.page_content for doc in retrieved_docs]
            eval_dict["retrieved_contexts"].append(docs_content)
            
            # Context Relevance 
            try:
                rel_result = await context_relevance_metrics(q, docs_content)
                rel_score = getattr(rel_result, 'value', getattr(rel_result, 'score', float(rel_result)))
            except Exception as e:
                print(f"Context Relevance 스코어 계산 실패: {e}")
                rel_score = 0.0
            context_relevance_scores.append(rel_score)
            
        mean_context_relevance = np.mean(context_relevance_scores) if context_relevance_scores else 0.0

        print("Ragas Metric 평가를 시작합니다 (Context Precision, Context Recall)...")
        eval_dataset_ragas = Dataset.from_dict(eval_dict)
        
        # Ragas evaluate 호출
        result = evaluate(
            dataset=eval_dataset_ragas,
            metrics=[context_precision, context_recall]
        )
        
        # 평가 결과 추출
        result_df = result.to_pandas()
        mean_precision = result_df["context_precision"].mean() if "context_precision" in result_df.columns else 0.0
        mean_recall = result_df["context_recall"].mean() if "context_recall" in result_df.columns else 0.0

        print(f"결과 - Chunk [{chunk_size}] | CR(Relevance): {mean_context_relevance:.4f} | Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f}")

        all_results.append({
            "chunk_size": chunk_size,
            "mean_relevance": float(mean_context_relevance),
            "mean_precision": float(mean_precision),
            "mean_recall": float(mean_recall),
        })

    with open(DATA_DIR / "chunking_results_v4_baseline.txt", "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(f"Chunk Size: {r['chunk_size']}, Relevance: {r['mean_relevance']:.4f}, Precision: {r['mean_precision']:.4f}, Recall: {r['mean_recall']:.4f}\n")

    # Save results
    with open(DATA_DIR / "chunking_results_v4_baseline.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\nExperiment complete. Results saved to chunking_results_v4_baseline.json")

if __name__ == "__main__":
    asyncio.run(main())



    

    

    