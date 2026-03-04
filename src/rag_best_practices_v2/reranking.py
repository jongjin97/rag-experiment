import sys
import os
import asyncio
import numpy as np
import pandas as pd
from typing import List

import torch
from sentence_transformers import CrossEncoder

from datasets import Dataset

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance
from openai import AsyncOpenAI

from src.rag_best_practices_v2.retrieval import get_ensemble_retriever, context_relevance_metrics
from src.rag_best_practices_v2.hyde_experiment import HyDERetriever
from src.config import DATA_DIR, MODEL_NAME

# ----------------- Reranker 정의 -----------------
class BGEReranker:
    """
    DLM Reranking (Cross-Encoder)
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None): # 한국어/다국어 Reranking 에 특화된 모델
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading BGE Reranker ({model_name}) on {device}...")
        self.model = CrossEncoder(model_name, device=device, tokenizer_args={"padding": True, "truncation": True, "max_length": 512})
        
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []
            
        # [ [query, doc1_text], [query, doc2_text], ... ] 구조로 페어 생성
        pairs = [[query, doc.page_content] for doc in documents]
        
        # OOM(Out of Memory) 방지를 위해 batch_size를 명시적으로 낮춥니다. (기본 32 -> 4 또는 8)
        scores = self.model.predict(pairs, batch_size=8)
        
        # 내림차순 정렬
        sorted_indices = np.argsort(scores)[::-1]
        
        # top_k 추출 및 score 메타데이터에 기록
        top_docs = []
        for i in sorted_indices[:top_k]:
            doc = documents[i].copy()
            doc.metadata["rerank_score"] = float(scores[i])
            top_docs.append(doc)
            
        return top_docs

class RerankingRetriever(BaseRetriever):
    """
    Base Retriever의 결과물을 받아서 Reranker 모델로 다시 랭킹을 매기는 래퍼 리트리버
    """
    base_retriever: BaseRetriever
    reranker: object
    top_k: int = 5
    
    # Pydantic v1 Base모델일 수 있으므로 Class Config에 arbitrary type 허용
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Base Retriever를 통해 넉넉하게 문서를 가져옴 (내부 k 파라미터가 더 크면 좋지만 여기서는 동일하게 수행 후 리랭킹)
        # Note: 실제 환경에서는 Base Retriever k는 10~20으로 올리고, Reranker top_k를 5로 좁히는 것이 성능이 좋습니다.
        initial_docs = self.base_retriever.invoke(query)
        
        # 2. Reranker로 점수 부여 후 정렬, 상위 top_k 추출
        reranked_docs = self.reranker.rerank(query, initial_docs, self.top_k)
        
        return reranked_docs


async def main():
    print("🔍 v2 Reranking 성능 비교 실험 시작 (Baseline: HyDE + Hybrid)")
    eval_file_path = DATA_DIR / "eval_dataset_v3_merged.csv"
    
    if not eval_file_path.exists():
        print(f"평가 데이터셋을 찾을 수 없습니다: {eval_file_path}")
        return
        
    eval_df = pd.read_csv(eval_file_path, encoding="utf-8-sig")
    
    questions = eval_df.get("user_input", eval_df.get("question", pd.Series())).tolist()
    references = eval_df.get("reference", eval_df.get("ground_truth", pd.Series())).tolist()
    
    if not questions or not references:
        print("질문 또는 정답 컬럼이 없습니다.")
        return

    test_size = len(questions)
    print(f"총 {test_size}개의 쿼리에 대해 실험합니다.")

    # 1. Base Retriever 선언
    # get_ensemble_retriever 함수는 내부적으로 서브 리트리버를 초기화하므로
    # 호출 후 내부 bm25와 dense의 k값을 20으로 강제 변경합니다.
    hybrid_retriever = get_ensemble_retriever(alpha=0.5)
    for retriever in hybrid_retriever.retrievers:
        if hasattr(retriever, 'k'):
            retriever.k = 20
            print(f"k값 변경: {retriever.k}")
        elif hasattr(retriever, 'search_kwargs') and 'k' in retriever.search_kwargs:
            retriever.search_kwargs['k'] = 20
            print(f"k값 변경: {retriever.search_kwargs['k']}")
    
    # Hybrid 리트리버를 넉넉한 갯수로 셋팅 (BM25와 Dense 각각에서 더 많이 가져오도록 k 조정)
    
    hyde_hybrid_retriever = HyDERetriever(base_retriever=hybrid_retriever)
    
    # 2. Reranker 초기화
    reranker = BGEReranker()
    reranking_retriever = RerankingRetriever(
        base_retriever=hyde_hybrid_retriever, 
        reranker=reranker, 
        top_k=5
    )

    strategies = {
        "HyDE + Hybrid + Reranking": reranking_retriever
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
        for i, q in enumerate(questions):
            retrieved_docs = retriever_func.invoke(q)
            docs_content = [doc.page_content for doc in retrieved_docs]
            eval_dict["retrieved_contexts"].append(docs_content)
            
            # Context Relevance 수동 측정 (스코어 객체의 숫자 뽑기)
            try:
                rel_result = await context_relevance_metrics(q, docs_content)
                rel_score = getattr(rel_result, 'value', getattr(rel_result, 'score', float(rel_result)))
            except Exception as e:
                print(f"CR 에러: {e}")
                rel_score = 0.0
            context_relevance_scores.append(rel_score)
            
            if (i+1) % 10 == 0 or (i+1) == test_size:
                print(f" - 진행: {i+1}/{test_size} (수동 CR 평가 완료)")

        print("\n⏳ Ragas 내부 evaluate() 호출 (Context Precision / Recall)...")
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
    print("\n\n🏆 Reranking 비교 평가 최종 결과")
    df_results = pd.DataFrame(all_results)
    print(df_results)
    df_results.to_csv(DATA_DIR / "experiment_retrieval_reranking.csv", index=False)
    print("\n결과가 data/experiment_retrieval_reranking.csv 에 저장되었습니다.")
    
if __name__ == "__main__":
    asyncio.run(main())
