import sys
import os
import asyncio
import pandas as pd
from typing import List

# Google Colab 환경에서의 경로 동기화 처리를 위한 구문
COLAB_BASE_PATH = "/content/drive/MyDrive/rag"
if COLAB_BASE_PATH not in sys.path:
    sys.path.append(COLAB_BASE_PATH)

# OOM 방지를 위한 메모리 할당 환경변수 셋팅 (import torch 이전에 수행)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

from src.config import DATA_DIR
from src.rag_best_practices_v2.retrieval import get_ensemble_retriever
from src.rag_best_practices_v2.hyde_experiment import HyDERetriever
from src.rag_best_practices_v2.reranking import BGEReranker

async def main():
    print("🔍 [Colab Debug] 0점 케이스 데이터 상세 분석 스크립트")
    
    # 0점 추출 데이터셋 경로
    eval_file_path = DATA_DIR / "experiment_generation_zero_scores.csv"
    
    if not eval_file_path.exists():
        print(f"평가 데이터셋을 찾을 수 없습니다: {eval_file_path}")
        return
        
    eval_df = pd.read_csv(eval_file_path, encoding="utf-8-sig")
    
    # 질문과 정답 (결측치 등 예외처리 포함)
    questions = eval_df.get("user_input", eval_df.get("question", pd.Series())).tolist()
    references = eval_df.get("reference", eval_df.get("ground_truth", pd.Series())).tolist()
    
    if not questions or not references:
        print("질문 또는 정답 컬럼이 없습니다.")
        return

    test_size = len(questions)
    print(f"총 {test_size}개의 0점 쿼리에 대해 디버그 분석을 시작합니다.")

    # 1. Base Retriever 파이프라인 셋팅 (Baseline: HyDE + Hybrid)
    print("Retriever 파이프라인 셋팅 중...")
    hybrid_retriever = get_ensemble_retriever(alpha=0.5)
    
    # k값 (검색량) 설정
    for retriever in hybrid_retriever.retrievers:
        if hasattr(retriever, 'k'):
            retriever.k = 20
        elif hasattr(retriever, 'search_kwargs') and 'k' in retriever.search_kwargs:
            retriever.search_kwargs['k'] = 20
            
    hyde_hybrid_retriever = HyDERetriever(base_retriever=hybrid_retriever)
    
    # 2. Reranker의 경우 GPU 환경(Colab)에서 더욱 빠르게 동작합니다
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reranker: 장치 구성 확인 -> {device}")
    reranker = BGEReranker(device=device)

    # 디버깅 결과를 저장할 리스트
    debug_results = []

    print("검색 및 Reranking 분석 진행 중...")
    for i, q in enumerate(questions):
        try:
            # 1. Base 검색 (HyDE + Hybrid) - 초기 검색 문서 전체 (넓은 범위)
            initial_docs = hyde_hybrid_retriever.invoke(q)
            
            # 2. Reranking (Cross-Encoder) - 순위 재배치 및 Top-5 추출
            reranked_docs = reranker.rerank(q, initial_docs, top_k=5)
            
            # 내용을 정리하여 저장할 형태 구성
            initial_contents = [f"[Initial Doc {idx+1}]\n{doc.page_content}" for idx, doc in enumerate(initial_docs)]
            
            # Reranker의 점수는 metadata["rerank_score"] 에 저장됨
            reranked_contents = []
            for idx, doc in enumerate(reranked_docs):
                score = doc.metadata.get("rerank_score", "N/A")
                if isinstance(score, float):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)
                reranked_contents.append(f"[Rank {idx+1} | Score: {score_str}]\n{doc.page_content}")
                
            debug_results.append({
                "user_input": q,
                "reference": references[i],
                "initial_retrieved_count": len(initial_docs),
                "initial_search_contexts": "\n\n=== [NEXT DOC] ===\n\n".join(initial_contents),
                "reranked_contexts": "\n\n=== [NEXT DOC] ===\n\n".join(reranked_contents)
            })
        except Exception as e:
            print(f"Error on query {i+1}: {e}")
            debug_results.append({
                "user_input": q,
                "reference": references[i],
                "initial_retrieved_count": 0,
                "initial_search_contexts": f"ERROR: {str(e)}",
                "reranked_contexts": f"ERROR: {str(e)}"
            })
            
        # 반복문이 돌 때마다 GPU 메모리 누수를 방지하기 위해 캐시 강제 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if (i+1) % 5 == 0 or (i+1) == test_size:
            print(f" - 진행: {i+1}/{test_size} 완료")

    # 3. 결과 저장
    result_df = pd.DataFrame(debug_results)
    
    # Ragas 평가 점수 등 메타정보를 함께 보고 싶다면 기존 df와 병합(merge)
    # 기존 데이터의 결과들을 보존하여 비교하기 위함
    final_df = pd.merge(eval_df, result_df, on=["user_input", "reference"], how="left")
    
    output_path = DATA_DIR / "experiment_zero_scores_debug_results.csv"
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n==============================================")
    print(f"✅ 디버그용 추출이 완료되었습니다.")
    print(f"저장 경로: {output_path}")
    print(f"==============================================")

if __name__ == "__main__":
    asyncio.run(main())
