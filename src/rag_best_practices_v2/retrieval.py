import sys
import os
from pathlib import Path

try:
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content"):
        sys.path.append("/content/drive/MyDrive/rag")
except Exception:
    pass

import json
import asyncio
import numpy as np
import pandas as pd
from typing import List, Optional

from datasets import Dataset

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_classic.retrievers import EnsembleRetriever

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance

from src.config import CHROMA_DB_DIR, DATA_DIR, MODEL_NAME

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
TEST_CHUNK_SIZE = 512

class CachedEmbeddings(Embeddings):
    """
    한 번 임베딩된 쿼리를 메모리에 저장(캐싱)하여 중복 API 호출 및 연산 시간을 방지하는 래퍼(Wrapper) 
    """
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings
        self.cache = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 평가지표 테스트 중에는 문서를 대량으로 새로 임베딩할 일이 없으므로 기본 로직 통과
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        # 캐시에 없으면 모델을 돌리고, 있으면 캐시에서 바로 0초 만에 꺼내줌
        if text not in self.cache:
            self.cache[text] = self.base_embeddings.embed_query(text)
        return self.cache[text]

_global_embeddings = None

def get_embeddings():
    global _global_embeddings
    if _global_embeddings is None:
        print(f"🔹 Embedding 모델 로드 및 캐시 생성: {EMBEDDING_MODEL_NAME}")
        base_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        _global_embeddings = CachedEmbeddings(base_emb)
    return _global_embeddings

def get_vectorstore(chunk_size: int = TEST_CHUNK_SIZE):
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=get_embeddings(),
        collection_name=f"samsung_chunking_{chunk_size}_v7",
    )

def get_dense_retriever():
    print("🔹 Dense (Vector) Retriever 초기화 중...")
    vectorstore = get_vectorstore(TEST_CHUNK_SIZE)
    if vectorstore._collection.count() == 0:
        raise ValueError("ChromaDB 컬렉션이 비어있습니다. build_chromadb.py를 먼저 실행하세요.")
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_bm25_retriever():
    print("🔹 BM25 (Keyword) Retriever 초기화 중...")
    vectorstore = get_vectorstore(TEST_CHUNK_SIZE)
    result = vectorstore.get() 
    
    texts = result['documents']
    metadatas = result['metadatas']
    
    docs = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    print(f"BM25 백그라운드용 문서 {len(docs)}개 로드 완료.")
    
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 5
    return retriever

def get_ensemble_retriever(alpha: float = 0.5):
    print(f"🔹 Ensemble (Hybrid) Retriever 초기화 중... (BM25: {alpha}, Dense: {1-alpha})")
    bm25 = get_bm25_retriever()
    dense = get_dense_retriever()
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[alpha, 1.0 - alpha]
    )
    return ensemble_retriever



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
    print("🔍 v2 Retrieval 평가 파이프라인 시작")
    eval_file_path = DATA_DIR / "eval_dataset_v3_merged.csv"
    
    if not eval_file_path.exists():
        print(f"평가 데이터셋을 찾을 수 없습니다: {eval_file_path}")
        return
        
    eval_df = pd.read_csv(eval_file_path, encoding="utf-8-sig")
    
    questions = eval_df.get("user_input", eval_df.get("question", pd.Series())).tolist()
    references = eval_df.get("reference", eval_df.get("ground_truth", pd.Series())).tolist()
    
    if not questions or not references:
        print("질문(user_input, question) 또는 정답(reference, ground_truth) 컬럼이 없습니다.")
        return

    # 전체 데이터셋을 평가에 사용합니다.
    test_size = len(questions)
    print(f"총 {test_size}개의 전체 쿼리에 대해 실험합니다.")

    strategies = {
        "Dense": get_dense_retriever(),
        "BM25": get_bm25_retriever(),
        "Hybrid": get_ensemble_retriever(alpha=0.5)
    }
    
    all_results = []
    
    for strategy_name, retriever_func in strategies.items():
        print(f"\n==============================================")
        print(f"🚀 실행 전략: {strategy_name}")
        print(f"==============================================")
        
        eval_dict = {
            "user_input": questions,
            "retrieved_contexts": [],
            "reference": references
        }
        
        context_relevance_scores = []
        for q in questions:
            retrieved_docs = retriever_func.invoke(q)
                
            docs_content = [doc.page_content for doc in retrieved_docs]
            eval_dict["retrieved_contexts"].append(docs_content)
            
            # Context Relevance (Slow API Request)
            try:
                rel_result = await context_relevance_metrics(q, docs_content)
                rel_score = getattr(rel_result, 'value', getattr(rel_result, 'score', float(rel_result)))
            except Exception as e:
                print(f"CR 에러: {e}")
                rel_score = 0.0
            context_relevance_scores.append(rel_score)

        print("\n⏳ Ragas 내부 evaluate() 를 호출합니다 (Context Precision/Recall 측정)...")
        # evaluate 내부적으로 EventLoop(asyncio)가 동작합니다 
        # Ragas의 evaluate는 동기함수인데 병렬적으로 태스크를 쪼개 돌려줍니다.
        eval_dataset_ragas = Dataset.from_dict(eval_dict)
        result = evaluate(
            dataset=eval_dataset_ragas,
            metrics=[context_precision, context_recall]
        )
        
        result_df = result.to_pandas()
        mean_precision = result_df["context_precision"].mean() if "context_precision" in result_df.columns else 0.0
        mean_recall = result_df["context_recall"].mean() if "context_recall" in result_df.columns else 0.0
        mean_relevance = np.mean(context_relevance_scores) if context_relevance_scores else 0.0

        print(f">>> [결과] {strategy_name} | CR: {mean_relevance:.4f} | Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f}")
        
        all_results.append({
            "Strategy": strategy_name,
            "Relevance": float(mean_relevance),
            "Precision": float(mean_precision),
            "Recall": float(mean_recall)
        })

    # 최종 병합 출력
    print("\n\n🏆 전략별 최종 비교 결과")
    df_results = pd.DataFrame(all_results)
    print(df_results)
    df_results.to_csv(DATA_DIR / "experiment_retrieval_v4.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
