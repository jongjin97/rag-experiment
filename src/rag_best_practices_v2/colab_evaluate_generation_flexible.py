import sys
import os
import asyncio
import numpy as np
import pandas as pd
from typing import List

# Google Colab 환경에서의 경로 동기화 처리를 위한 구문
COLAB_BASE_PATH = "/content/drive/MyDrive/rag"
if COLAB_BASE_PATH not in sys.path:
    sys.path.append(COLAB_BASE_PATH)

# OOM 방지를 위한 메모리 할당 환경변수 셋팅 (import torch 이전에 수행)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import Dataset

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# sys.path가 올바르게 잡혀 있어야 프로젝트 내부 모듈들을 정상적으로 임포트할 수 있습니다.
from src.config import DATA_DIR, MODEL_NAME
from src.rag_best_practices_v2.retrieval import get_ensemble_retriever
from src.rag_best_practices_v2.hyde_experiment import HyDERetriever
from src.rag_best_practices_v2.reranking import BGEReranker, RerankingRetriever

async def generate_answer(query: str, contexts: List[str], qa_chain) -> str:
    # 여러 Context를 하나의 문자열로 결합
    context_str = "\n\n".join(contexts)
    result = await qa_chain.ainvoke({"context": context_str, "question": query})
    return result

async def main():
    print("🔍 [Colab Version] v2 Generation 성능 평가 시작 (과도한 보수성 완화/유연한 프롬프트 적용)")
    
    # Colab 내 데이터셋 경로 확인
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
    # NOTE: 시간이 너무 오래 걸린다면 아래 주석을 풀고 샘플 테스트할 수 있습니다.
    # test_size = min(len(questions), 15)
    # questions = questions[:test_size]
    # references = references[:test_size]

    print(f"총 {test_size}개의 쿼리에 대해 실험합니다.")

    # 1. Retriever 파이프라인 구성
    print("Retriever 파이프라인 셋팅 중...")
    hybrid_retriever = get_ensemble_retriever(alpha=0.5)
    
    for retriever in hybrid_retriever.retrievers:
        if hasattr(retriever, 'k'):
            retriever.k = 20
        elif hasattr(retriever, 'search_kwargs') and 'k' in retriever.search_kwargs:
            retriever.search_kwargs['k'] = 20
            
    hyde_hybrid_retriever = HyDERetriever(base_retriever=hybrid_retriever)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reranker: 장치 구성 확인 -> {device}")
    reranker = BGEReranker(device=device)
    reranking_retriever = RerankingRetriever(
        base_retriever=hyde_hybrid_retriever, 
        reranker=reranker, 
        top_k=5
    )

    # 2. Generation 파이프라인 (답변 생성기) 셋팅: ★프롬프트 개선★
    print("LLM Generation 파이프라인 셋팅 중...")
    gen_model_name = MODEL_NAME if "gpt" in MODEL_NAME else "gpt-4o-mini"
    llm = ChatOpenAI(model=gen_model_name, temperature=0.0)
    
    prompt = ChatPromptTemplate.from_template("""
당신은 삼성전자 문서 분석 전문가입니다.
아래 제공된 [Context] 정보만을 바탕으로 [Question]에 대한 답변을 한국어로 정확하게 작성하세요.

[중요 지시사항 - 유연한 문맥 활용]
만약 [Question]이 묻는 전체 정보를 완벽히 포괄하는 내용이 없더라도, [Context] 내에 질문과 관련된 "부분적인 힌트(특정 국가, 특정 연도 등 부분적 정보)"가 존재한다면 그 내용을 토대로 최대한 성실하게 유연한 답변을 제공하세요.

단, 관련된 정보나 힌트조차 아예 단 한 줄도 없다면 억지로 지어내지 말고 "제공된 문서에서 정보를 찾을 수 없습니다."라고 답변하세요.

[Context]
{context}

[Question]
{question}

Answer:
""")
    qa_chain = prompt | llm | StrOutputParser()

    # Ragas 0.4.x 평가 딕셔너리
    eval_dict = {
        "user_input": questions,
        "retrieved_contexts": [],
        "response": [],
        "reference": references
    }

    # 3. 검색 및 답변 생성
    print("검색 및 답변 생성 진행 중...")
    for i, q in enumerate(questions):
        retrieved_docs = reranking_retriever.invoke(q)
        docs_content = [doc.page_content for doc in retrieved_docs]
        
        generated_answer = await generate_answer(q, docs_content, qa_chain)
        
        eval_dict["retrieved_contexts"].append(docs_content)
        eval_dict["response"].append(generated_answer)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if (i+1) % 10 == 0 or (i+1) == test_size:
            print(f" - 진행: {i+1}/{test_size} 완료")

    # 4. Ragas 평가
    print("\n⏳ Ragas 내부 evaluate() 호출 (Answer Relevancy / Faithfulness 측정 중)...")
    dataset = Dataset.from_dict(eval_dict)
    
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=gen_model_name, temperature=0.0))
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    answer_rel_metric = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
    faithfulness_metric = Faithfulness(llm=eval_llm)
    
    result = evaluate(
        dataset=dataset,
        metrics=[answer_rel_metric, faithfulness_metric]
    )
    
    # 5. 결과 요약 및 저장
    result_df = result.to_pandas()
    
    mean_answer_relevancy = result_df["answer_relevancy"].mean() if "answer_relevancy" in result_df.columns else 0.0
    mean_faithfulness = result_df["faithfulness"].mean() if "faithfulness" in result_df.columns else 0.0

    print(f"\n==============================================")
    print(f"🏆 유연한 프롬프트 Generation 평가 최종 결과 (Colab)")
    print(f"Answer Relevancy: {mean_answer_relevancy:.4f}")
    print(f"Faithfulness    : {mean_faithfulness:.4f}")
    print(f"==============================================")
    
    output_path = DATA_DIR / "experiment_generation_metrics_flexible.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n결과가 {output_path} 파일로 저장되었습니다.")

if __name__ == "__main__":
    import torch
    asyncio.run(main())
