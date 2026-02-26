import os
import re
import json
import asyncio
import pandas as pd
import openai
from pathlib import Path

# Ragas 0.4.x imports
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, HeadlinesExtractor, KeyphrasesExtractor
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DATA_DIR, MODEL_NAME

# -------------------------
# Configuration
# -------------------------
TEST_SIZE = 50
TARGET_DOC_DIR_NAME = "[삼성전자] 사업보고서(일반법인) (2025.03.11)"
BASE_PATH = DATA_DIR / "processed_experiment" / TARGET_DOC_DIR_NAME
TEXT_FILE = BASE_PATH / "document_text_with_placeholders.txt"
TABLES_FILE = BASE_PATH / "extracted_tables.json"
OUTPUT_FILE = DATA_DIR / "eval_dataset_v3_4.csv"

# gen_dataset.py 와 동일한 Personas 설정
personas = [
    Persona(
        name="Financial Analyst",
        role_description="""
        당신은 삼성전자의 사업보고서에 기재된 '구체적인 수치'와 '재무 데이터'를 확인하는 깐깐한 재무 분석가입니다.
        
        [Strict Criteria]
        1. Fact-Based: 반드시 텍스트에 명시적으로 적혀 있는 숫자나 공식적인 재무 상태에 대해서만 질문하십시오.
        2. No Speculation: 텍스트에 없는 미래 전망이나 재무적 영향을 추측하여 질문하지 마십시오.
        3. Language: 모든 질문과 답변은 완벽한 한국어(Korean)여야 합니다.
        """
    ),
    Persona(
        name="Fact Checker",
        role_description="""
        당신은 보고서의 '객관적 사실'만을 검증하는 팩트 체커입니다. '이미 일어난 사건'에 집중합니다.
        
        [Strict Criteria]
        1. Verification: "출시했다", "개발했다", "수상했다" 등 과거 완료형 팩트에 대해서만 질문하십시오.
        2. No Buzzwords: "잠재력", "시너지", "기대효과" 같은 추상적인 단어가 포함된 질문은 생성하지 마십시오.
        3. Language: 모든 질문과 답변은 완벽한 한국어(Korean)여야 합니다.
        """
    ),
    Persona(
        name="Compliance Officer",
        role_description="""
        당신은 보고서에 '명시적으로 언급된' 법적 소송, 제재 현황만을 확인하는 준법 감시인입니다.
        
        [Strict Criteria]
        1. Explicit Risk Only: 텍스트에 "소송", "제재", "벌금", "위반" 단어가 없다면 리스크 관련 질문을 절대 만들지 마십시오.
        2. No Inference: 단순 주소 변경이나 합병을 비즈니스 리스크로 확대 해석하지 마십시오.
        3. No Unanswerable: 문서에 없는 내용을 묻고 "자료에 없습니다"라고 자체 결론 짓는 질문을 절대 생성하지 마십시오.
        4. Language: 모든 질문과 답변은 완벽한 한국어(Korean)여야 합니다.
        """
    )
]

def get_kg(docs):
    """Langchain Documents를 Ragas KnowledgeGraph 형식으로 변환"""
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
        )
    return kg

def get_transforms(llm):
    """
    Ragas 테스트셋 전처리기
    에러 발생 원인이었던 HeadlineSplitter를 제거하여, 이미 Chunking 완료된 텍스트가 다시 쪼개지다 오류나는 현상을 방지
    """
    headline_extractor = HeadlinesExtractor(llm=llm, max_num=3, max_token_limit=16000)
    keyphrase_extractor = KeyphrasesExtractor(llm=llm, max_token_limit=16000, max_num=3)
    return [headline_extractor, keyphrase_extractor]

async def generate_dataset_v3():
    # -------------------------
    # 1. 전처리된 문서(Placeholder 보존) 로드
    # -------------------------
    print(f"Loading preprocessed text from {TEXT_FILE}...")
    if not TEXT_FILE.exists() or not TABLES_FILE.exists():
        print(f"Error: Required files not found in {BASE_PATH}")
        return

    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        full_text = f.read()
        
    documents = [Document(page_content=full_text, metadata={"source": TARGET_DOC_DIR_NAME})]

    # -------------------------
    # 2. 문서 분할 (필수)
    # -------------------------
    print("Splitting documents via Langchain...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)
    
    # 너무 짧은 청크는 HeadlinesExtractor 실패 확률을 높이므로 필터링 (gen_dataset.py 방식 참고)
    docs = [doc for doc in docs if len(doc.page_content) >= 100]
    print(f"Documents split into {len(docs)} chunks (filtered short chunks).")

    # -------------------------
    # 2-1. Placeholder를 실제 Markdown 테이블로 누적 병합(치환)
    # [TABLE_0_1] 발견 시 TABLE_0_0과 TABLE_0_1을 모두 합칩니다.
    # -------------------------
    print("Replacing table placeholders with Markdown...")
    with open(TABLES_FILE, 'r', encoding='utf-8') as f:
        tables = json.load(f)
        
    pattern = re.compile(r'\[TABLE_(\d+)_(\d+)\]')
    
    for doc in docs:
        matches = pattern.findall(doc.page_content)
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
                doc.page_content = doc.page_content.replace(placeholder, f"\n\n{full_table_md}\n\n")
            else:
                doc.page_content = doc.page_content.replace(placeholder, "")

    # -------------------------
    # 3. LLM & Embedding 설정
    # -------------------------
    print("Initializing LLM and Embeddings...")
    model_to_use = MODEL_NAME if "gpt" in MODEL_NAME else "gpt-4o-mini"
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_to_use))
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    # -------------------------
    # 4. Knowledge Graph 수동 구축 및 Transform 적용 (gen_dataset.py 방식)
    # -------------------------
    print("Creating Knowledge Graph and applying specific Transforms...")
    kg = get_kg(docs)
    transforms = get_transforms(generator_llm)
    # apply_transforms 가 Headlines 와 Keyphrases 만 안전하게 추출함
    apply_transforms(kg, transforms)

    # 억지 질문 방지를 위한 Synthesizer 세팅
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"), 0.5),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.5)
    ]

    print("Setting up Testset Generator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas
    )

    # -------------------------
    # 5. 데이터셋 생성 (docs를 바로 넣지 않음)
    # -------------------------
    print(f"Generating {TEST_SIZE} query-reference pairs...")
    testset = generator.generate(
        testset_size=TEST_SIZE,
        query_distribution=query_distribution
    )

    # -------------------------
    # 6. DataFrame 변환 및 이상치 검열
    # -------------------------
    df = testset.to_pandas()
    
    def is_valid_qa(row):
        ref = str(row.get('reference', '')) 
        user_input = str(row.get('user_input', ''))
        
        # DataFrame 컬럼명이 버전별로 약간 다를 경우를 대비한 폴백 처리
        if not ref and 'ground_truth' in row:
             ref = str(row.get('ground_truth', ''))
        if not user_input and 'question' in row:
             user_input = str(row.get('question', ''))

        bad_phrases = [
            "포함되어 있지 않", "알 수 없", "언급되지 않았", 
            "해당사항 없", "파악할 수 없", "명시되어 있지 않"
        ]
        if any(phrase in ref for phrase in bad_phrases):
            return False
            
        if len(user_input) < 10:
            return False
        return True

    original_len = len(df)
    df = df[df.apply(is_valid_qa, axis=1)]
    filtered_count = original_len - len(df)
    if filtered_count > 0:
        print(f"🚨 [필터링 작동] 억지 질문/안전 회피성 답변 {filtered_count}건이 제거되었습니다.")

    print("\n--- [First 5 generated rows] ---")
    print(df.head())

    # -------------------------
    # 7. 저장
    # -------------------------
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("Done!")

if __name__ == "__main__":
    asyncio.run(generate_dataset_v3())
