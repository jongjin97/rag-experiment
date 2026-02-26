import os
import re
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List

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

def load_all_processed_documents():
    """전처리된 마크다운 플레이스홀더 텍스트와 테이블 정보를 [삼성전자]로 시작하는 모든 폴더에서 로드합니다."""
    base_dir = DATA_DIR / "processed_experiment"
    all_documents = []
    
    for dir_path in base_dir.iterdir():
        if not dir_path.is_dir() or not dir_path.name.startswith("[삼성전자]"):
            continue
            
        text_file = dir_path / "document_text_with_placeholders.txt"
        tables_file = dir_path / "extracted_tables.json"
        
        if not text_file.exists() or not tables_file.exists():
            print(f"경고: {dir_path.name} 폴더에 필요한 파일이 없습니다. 스킵합니다.")
            continue
            
        with open(text_file, 'r', encoding='utf-8') as f:
            full_text = f.read()
        with open(tables_file, 'r', encoding='utf-8') as f:
            tables = json.load(f)
            
        doc = Document(page_content=full_text, metadata={"source": dir_path.name})
        all_documents.append((doc, tables))
        print(f"문서 로드 완료: {dir_path.name}")
        
    return all_documents

def get_embeddings():
    """Embedding 모델을 초기화하고 반환합니다."""
    print(f"Embedding 모델 로딩 중: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore(chunk_size):
    """ChromaDB Vector Store를 초기화하고 반환합니다."""
    embedding_function = get_embeddings()
    # persistent_client=None을 설정하면 Chroma가 persist_directory를 통해 연결을 자동으로 관리합니다.
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_function,
        collection_name=f"samsung_chunking_{chunk_size}_v7",
    )

def split_documents_and_restore_tables(doc_tables_list: List[tuple], chunk_size: int = 1000, chunk_overlap: int = 20) -> List[Document]:
    """
    여러 문서를 각각 Chunk 단위로 먼저 분할한 뒤, 각 문서별 독립적인 테이블 데이터를 이용해 플레이스홀더를 치환합니다.
    분할 전 커스텀 length_function을 적용하여, 플레이스홀더가 치환되었을 때의 실제 텍스트 길이를 기반으로 청킹하게 합니다.
    """
    all_restored_chunks = []
    pattern = re.compile(r'\[TABLE_(\d+)_(\d+)\]')
    
    for doc, tables in doc_tables_list:
        
        def custom_length_function(text: str) -> int:
            length = len(text)
            matches = pattern.findall(text)
            for x_str, y_str in matches:
                placeholder_len = len(f"[TABLE_{x_str}_{y_str}]")
                length -= placeholder_len
                
                y = int(y_str)
                combined_markdown = []
                for i in range(y + 1):
                    chunk_id = f"TABLE_{x_str}_{i}"
                    if chunk_id in tables:
                        chunk_md = tables[chunk_id].get('markdown', '')
                        if chunk_md:
                            combined_markdown.append(chunk_md)
                
                if combined_markdown:
                    full_table_md = "\n\n".join(combined_markdown)
                    length += len(f"\n\n{full_table_md}\n\n")
            return length
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=custom_length_function,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = text_splitter.split_documents([doc])
        
        # Placeholder 치환 작업 (TABLE_X_Y 형태일 경우 TABLE_X_0 부터 TABLE_X_Y 까지 누적 합체)
        for chunk_doc in docs:
            matches = pattern.findall(chunk_doc.page_content)
            for x_str, y_str in set(matches):
                placeholder = f"[TABLE_{x_str}_{y_str}]"
                y = int(y_str)
                
                combined_markdown = []
                for i in range(y + 1):
                    chunk_id = f"TABLE_{x_str}_{i}"
                    if chunk_id in tables:
                        chunk_md = tables[chunk_id].get('markdown', '')
                        if chunk_md:
                            combined_markdown.append(chunk_md)
                
                if combined_markdown:
                    full_table_md = "\n\n".join(combined_markdown)
                    chunk_doc.page_content = chunk_doc.page_content.replace(placeholder, f"\n\n{full_table_md}\n\n")
                else:
                    chunk_doc.page_content = chunk_doc.page_content.replace(placeholder, "")
            all_restored_chunks.append(chunk_doc)
            
    return all_restored_chunks


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

    # 전체 데이터셋을 사용하여 정확한 평가를 진행합니다.
    test_size = len(questions)
    print(f"총 {test_size}개의 전체 쿼리에 대해 평가를 진행합니다...")

    # 2. 플레이스홀더 적용 다중 원본 문서 로드
    doc_tables_list = load_all_processed_documents()
    print(f"총 {len(doc_tables_list)}개의 전처리된 문서를 로드했습니다.")

    chunk_sizes = [256, 512, 1024]
    chunk_overlap = 20
    
    all_results = []
    
    for chunk_size in chunk_sizes:
        print(f"\n=== Chunk 사이즈 처리 중: {chunk_size} ===")
        vectorstore = get_vectorstore(chunk_size)

        if vectorstore._collection.count() > 0:
            print(f"Chunk 사이즈 {chunk_size}에 대한 Vector Store가 이미 존재합니다. 추가 인덱싱을 스킵합니다...")
        else:
            # 여러 문서를 분할하고 각각의 테이블을 마크다운으로 환원하여 Vector Store에 추가
            documents = split_documents_and_restore_tables(doc_tables_list, chunk_size, chunk_overlap)
                
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                print(f"Batch 인덱싱 완료 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

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

    # 최종 결과 저장
    with open(DATA_DIR / "chunking_results_v3.txt", "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(f"Chunk Size: {r['chunk_size']}, Relevance: {r['mean_relevance']:.4f}, Precision: {r['mean_precision']:.4f}, Recall: {r['mean_recall']:.4f}\n")

    with open(DATA_DIR / "chunking_results_v3.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\n실험이 완료되었습니다. 결과가 chunking_results_v3.json에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())



    

    

    