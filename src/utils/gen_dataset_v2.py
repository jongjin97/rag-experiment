import os
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import apply_transforms, HeadlinesExtractor, KeyphrasesExtractor
from langchain_openai import ChatOpenAI
from src.config import DATA_DIR, MODEL_NAME

# Configuration
TEST_SIZE = 50
BATCH_SIZE = 5
# 타겟 디렉토리 설정 (추후 인자 처리 가능하도록 변수화)
TARGET_DOC_DIR_NAME = "[삼성전자] 반기보고서(일반법인) (2025.08.14)"
BASE_PATH = DATA_DIR / "processed_experiment" / TARGET_DOC_DIR_NAME
CHUNKS_FILE = BASE_PATH / "final_chunks.json"
TABLES_FILE = BASE_PATH / "extracted_tables.json"
OUTPUT_FILE = DATA_DIR / "eval_dataset_v2.json" # 출력 파일명 변경

# Define Personas (기존 gen_dataset.py와 동일)
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

def get_transforms(llm):
    """Knowledge Graph 노드에 메타데이터(Headline, Keyphrase)를 추가하기 위한 추출기들을 반환합니다."""
    headline_extractor = HeadlinesExtractor(llm=llm, max_num=3)
    keyphrase_extractor = KeyphrasesExtractor(llm=llm, max_num=3)
    return [headline_extractor, keyphrase_extractor]

def load_data(chunks_path: Path, tables_path: Path) -> tuple[List[str], Dict[str, Any]]:
    """JSON 파일에서 청크와 테이블 데이터를 로드합니다."""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
        
    return chunks, tables

def preprocess_chunks(chunks: List[str], tables: Dict[str, Any]) -> List[str]:
    """청크 내의 테이블 플레이스홀더를 실제 마크다운 테이블로 치환합니다."""
    processed_chunks = []
    for chunk in chunks:
        processed_text = chunk
        # 간단한 문자열 치환 사용 (정규식보다 빠르고 안전할 수 있음, 플레이스홀더 형식이 명확하므로)
        # 테이블 키를 순회하며 치환하는 것은 비효율적일 수 있으나, 테이블 수가 아주 많지 않다면 허용 가능.
        # 효율을 위해 청크 내에 '[' 문자가 있을 때만 검사할 수도 있음.
        if "[" in processed_text and "TABLE_" in processed_text:
            for table_id, table_data in tables.items():
                placeholder = f"[{table_id}]"
                if placeholder in processed_text:
                    markdown_table = table_data.get('markdown', '')
                    # 마크다운 테이블이 비어있지 않은 경우에만 치환, 앞뒤 개행 추가
                    if markdown_table:
                        processed_text = processed_text.replace(placeholder, f"\n{markdown_table}\n")
                    else:
                        processed_text = processed_text.replace(placeholder, "") # 내용 없으면 제거
        
        processed_chunks.append(processed_text)
    return processed_chunks

def create_knowledge_graph(chunks: List[str]) -> KnowledgeGraph:
    """전처리된 청크 리스트를 사용하여 KnowledgeGraph를 생성합니다."""
    kg = KnowledgeGraph()
    for chunk_text in chunks:
        # Node 생성 (metadata는 최소화, 필요 시 추가 가능)
        # chunk_overlap 정보가 없으므로 연결성은 떨어질 수 있음. 
        # 하지만 TestsetGenerator는 개별 노드 내용을 기반으로도 질문 생성 가능.
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": chunk_text,
                    "document_metadata": {"source": TARGET_DOC_DIR_NAME} 
                }
            )
        )
    return kg

async def generate_dataset_v2():
    print(f"Loading data from {BASE_PATH}...")
    if not CHUNKS_FILE.exists() or not TABLES_FILE.exists():
        print(f"Error: Required files not found in {BASE_PATH}")
        return

    chunks, tables = load_data(CHUNKS_FILE, TABLES_FILE)
    print(f"Loaded {len(chunks)} chunks and {len(tables)} tables.")

    print("Preprocessing chunks (replacing table placeholders)...")
    processed_chunks = preprocess_chunks(chunks, tables)
    
    # 예시 확인
    print(f"Sample processed chunk (first 100 chars): {processed_chunks[0][:100]}...")

    print("Creating Knowledge Graph...")
    kg = create_knowledge_graph(processed_chunks)

    print("Initializing Ragas TestsetGenerator...")
    # Force use gpt-4o (기존 설정 따름)
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_to_use = MODEL_NAME if "gpt" in MODEL_NAME else "gpt-4o"
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_to_use))
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    print("Applying transforms to Knowledge Graph (extracting headlines & keyphrases)...")
    transforms = get_transforms(generator_llm)
    apply_transforms(kg, transforms)

    # 단순 page_content 기반이 아닌 추출된 핵심어/제목 기반으로 질문을 생성하여 할루시네이션(억지 질문) 방지
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"), 0.5),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.5)
    ]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas
    )

    # === 배치 처리 로직 (기존 코드 재사용) ===
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_json(OUTPUT_FILE, orient="records")
            print(f"기존 파일 발견: {len(existing_df)}개의 데이터가 있습니다.")
        except ValueError:
            print("기존 파일 읽기 실패. 새로 시작합니다.")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    current_count = len(existing_df)
    
    if current_count >= TEST_SIZE:
        print(f"이미 목표 수량({TEST_SIZE}) 이상이 생성되어 있습니다. 종료합니다.")
        return

    print(f"총 {TEST_SIZE}개 생성을 목표로 배치 처리를 시작합니다. (현재: {current_count}개)")

    while current_count < TEST_SIZE:
        next_batch_size = min(BATCH_SIZE, TEST_SIZE - current_count)
        
        print(f"\n[Batch Processing] {current_count}/{TEST_SIZE} 완료. 이번에 {next_batch_size}개 생성 시도...")
        
        try:
            # 배치만큼 생성
            # query_distribution을 명시적으로 줄 때는 해당 리스트 사용.
            # page_content 기반이므로 단순화된 distribution 사용
            batch_testset = generator.generate(
                testset_size=next_batch_size,
                query_distribution=query_distribution,
            )
            
            batch_df = batch_testset.to_pandas()
            
            # --- 🚀 이상 데이터 필터링 로직 (회피성 답변, 대답 불가 데이터 제거) ---
            def is_valid_qa(row):
                ref = str(row.get('reference', ''))
                user_input = str(row.get('user_input', ''))
                # 답변 내용에 "답할 수 없다"는 식의 명백한 회피 문구가 있는지 확인
                bad_phrases = [
                    "포함되어 있지 않", "알 수 없", "언급되지 않았", 
                    "해당사항 없", "파악할 수 없", "명시되어 있지 않"
                ]
                if any(phrase in ref for phrase in bad_phrases):
                    return False
                # 질문이 너무 짧거나 이상한 경우 방지
                if len(user_input) < 10:
                    return False
                return True

            original_len = len(batch_df)
            batch_df = batch_df[batch_df.apply(is_valid_qa, axis=1)]
            filtered_count = original_len - len(batch_df)
            
            if filtered_count > 0:
                print(f"  🚨 [필터링 작동] 억지 질문/안전 회피성 답변 {filtered_count}건이 제거되었습니다.")
                
            if batch_df.empty:
                print("  ⚠️ 이번 배치는 모두 유효하지 않은 데이터입니다. 다시 시도합니다.")
                continue
            # -------------------------------------------------------------------

            if existing_df.empty:
                existing_df = batch_df
            else:
                existing_df = pd.concat([existing_df, batch_df], ignore_index=True)
            
            existing_df.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=4)
            
            current_count = len(existing_df)
            print(f"--> 성공! 현재까지 총 {current_count}개 저장 완료. (파일: {OUTPUT_FILE})")
            
        except Exception as e:
            print(f"Error during batch generation: {e}")
            import traceback
            traceback.print_exc()
            print("중단되었습니다.")
            break 

    print(f"\n최종 완료. 총 {len(existing_df)}개의 데이터가 {OUTPUT_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(generate_dataset_v2())
