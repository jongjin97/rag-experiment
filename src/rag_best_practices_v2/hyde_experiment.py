import sys
import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List

from datasets import Dataset

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance
from openai import AsyncOpenAI

# 기존 retrieval.py의 구현체 (EnsembleRetriever 등) 재사용
from src.rag_best_practices_v2.retrieval import get_ensemble_retriever, context_relevance_metrics
from src.config import DATA_DIR, MODEL_NAME

# ----------------- HyDE Retriever 정의 -----------------
# 1. LLM에게 유저의 질문을 주고 "가상의 이상적인 답변"을 미리 작성하게 합니다.
# 2. 유저의 원래 질문 + 가상의 답변을 하나로 묶어 VectorDB와 BM25에 검색합니다.
# 3. 질문의 키워드뿐 아니라, 답변이 가질 법한 키워드와 문맥까지 함께 매칭되므로 성능 향상을 기대할 수 있습니다.

hyde_prompt = PromptTemplate.from_template(
    "다음 질문에 답하는 짧고 관련성 있는 사실 기반의 글(가상의 답변)을 작성해 주세요. 문맥 검색을 위한 용도입니다.\n\n질문: {query}\n\n답변:"
)
# 답변 생성용 모델은 속도와 가성비를 위해 gpt-4o-mini를 사용하는 것이 일반적입니다.
hyde_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()

class HyDERetriever(BaseRetriever):
    base_retriever: BaseRetriever
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. 가상의 답변(Hypothetical Document) 생성
        hypothetical_doc = hyde_chain.invoke({"query": query})
        
        # 2. 질문과 가상 답변을 합쳐서 검색 쿼리로 사용 (가중치 효과)
        search_query = f"{query}\n\n{hypothetical_doc}"
        
        # 3. Base Retriever (여기서는 Hybrid) 에게 전달
        return self.base_retriever.invoke(search_query)


async def main():
    print("🔍 v2 HyDE + Hybrid 성능 비교 실험 시작")
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
    print(f"총 {test_size}개의 쿼리에 대해 (Hybrid) vs (HyDE + Hybrid) 비교를 진행합니다.")

    # 1. 기본 Hybrid (Ensemble) 리트리버 로드
    hybrid_retriever = get_ensemble_retriever(alpha=0.5)
    
    # 2. HyDE가 덧씌워진 Hybrid 리트리버
    hyde_hybrid_retriever = HyDERetriever(base_retriever=hybrid_retriever)

    strategies = {
        "Hybrid (Baseline)": hybrid_retriever,
        "HyDE + Hybrid": hyde_hybrid_retriever
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
            
            # Context Relevance 
            try:
                rel_result = await context_relevance_metrics(q, docs_content)
                rel_score = getattr(rel_result, 'value', getattr(rel_result, 'score', float(rel_result)))
            except Exception as e:
                print(f"CR 에러: {e}")
                rel_score = 0.0
            context_relevance_scores.append(rel_score)
            
            if (i+1) % 10 == 0 or (i+1) == test_size:
                print(f" - 진행: {i+1}/{test_size} (수동 CR 평가)")

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
    print("\n\n🏆 HyDE 비교 평가 최종 결과")
    df_results = pd.DataFrame(all_results)
    print(df_results)
    df_results.to_csv(DATA_DIR / "experiment_retrieval_hyde.csv", index=False)
    print("\n결과가 data/experiment_retrieval_hyde.csv 에 저장되었습니다.")
    
if __name__ == "__main__":
    asyncio.run(main())
