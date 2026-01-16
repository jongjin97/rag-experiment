import os
import asyncio
import openai
from pathlib import Path
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.persona import Persona
from src.utils.document_loader import load_documents, load_documents_merged
from src.config import DATA_DIR, MODEL_NAME
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

# Configuration
TEST_SIZE = 50
BATCH_SIZE = 5
OUTPUT_FILE = DATA_DIR / "eval_dataset_merged_1.json"
DOCUMENT_DIR = DATA_DIR / "samsung" / "[삼성전자] 사업보고서(일반법인) (2025.03.11).pdf"

# Define Personas for Samsung Report Analysis
personas = [
    Persona(
        name="Financial Analyst",
        role_description="삼성전자의 분기 및 연간 보고서를 분석하는 수석 재무 분석가입니다. 매출, 영업이익, 부채 비율 및 재무 안정성에 중점을 둡니다. 생성하는 모든 질문과 답변은 반드시 한국어(Korean)로 작성해야 합니다.",
    ),
    Persona(
        name="Tech Investor",
        role_description="삼성전자의 DX(Device eXperience) 및 DS(Device Solutions) 부문의 성장 잠재력을 찾는 투자자입니다. 시장 점유율, 신제품 출시, R&D 투자에 관심이 많습니다. 생성하는 모든 질문과 답변은 반드시 한국어(Korean)로 작성해야 합니다.",
    ),
    Persona(
        name="Risk Manager",
        role_description="사업 보고서에 언급된 잠재적 비즈니스 리스크, 규제 문제 및 공급망 문제를 식별하는 리스크 평가 전문가입니다. 생성하는 모든 질문과 답변은 반드시 한국어(Korean)로 작성해야 합니다.",
    )
]

def get_kg(docs):
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
    headline_extractor = HeadlinesExtractor(llm=llm, max_num=3, max_token_limit=16000)
    headline_splitter = HeadlineSplitter(max_tokens=500)
    keyphrase_extractor = KeyphrasesExtractor(llm=llm, max_token_limit=16000, max_num=3)
    transforms = [headline_extractor, headline_splitter, keyphrase_extractor]
    return transforms

async def generate_dataset_in_batches_merged_docs():
    print("Loading documents...")
    documents = load_documents_merged(DOCUMENT_DIR)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    documents = splitter.split_documents(documents)


    print(f"Loaded {len(documents)} pages. Using a subset for generation to save costs...")
    # Using specific pages to get variety (Reduced subset for stability)
    selected_docs = documents[10:40] 
    # + documents[100:105] 

    print("Initializing Ragas TestsetGenerator...")
    
    kg = get_kg(selected_docs)

    # Force use gpt-4o for dataset generation as it requires high reasoning and token limits
    # gpt-5-mini was hitting max_token limits (3k)
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Providing a fallback model if MODEL_NAME is experimental/invalid
    model_to_use = MODEL_NAME if "gpt" in MODEL_NAME else "gpt-4o"
    
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_to_use))
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    transforms = get_transforms(generator_llm)
    apply_transforms(kg, transforms)

    query_distibution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"), 0.5),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.5)
    ]

    # Use default distributions by not specifying query_distribution manually
    # The generator will use default evolutions (simple, multi_context, reasoning, etc.)
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas
    )

    # === 배치 처리 로직 시작 ===
    
    # 기존 파일이 있다면 로드하여 이어쓰기 준비
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
        # 남은 개수 계산 (배치 사이즈와 남은 개수 중 작은 값 선택)
        next_batch_size = min(BATCH_SIZE, TEST_SIZE - current_count)
        
        print(f"\n[Batch Processing] {current_count}/{TEST_SIZE} 완료. 이번에 {next_batch_size}개 생성 시도...")
        
        try:
            # 배치만큼 생성
            batch_testset = generator.generate(
                testset_size=next_batch_size,
                query_distribution=query_distibution,
            )
            
            batch_df = batch_testset.to_pandas()
            
            # 데이터프레임 병합
            if existing_df.empty:
                existing_df = batch_df
            else:
                existing_df = pd.concat([existing_df, batch_df], ignore_index=True)
            
            # 중간 저장 (덮어쓰기)
            existing_df.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=4)
            
            current_count = len(existing_df)
            print(f"--> 성공! 현재까지 총 {current_count}개 저장 완료. (파일: {OUTPUT_FILE})")
            
        except Exception as e:
            print(f"Error during batch generation: {e}")
            print("중단되었습니다. 해결 후 다시 실행하면 이어서 진행됩니다.")
            break 

    print(f"\n최종 완료. 총 {len(existing_df)}개의 데이터가 {OUTPUT_FILE}에 저장되었습니다.")

async def generate_dataset_in_batches():
    print("Loading documents...")
    documents = load_documents(DOCUMENT_DIR)
    
    # Filter out documents that are too short (less than 100 characters)
    if documents:
        original_count = len(documents)
        documents = [doc for doc in documents if len(doc.page_content) >= 100]
        print(f"Filtered out {original_count - len(documents)} short documents. Remaining: {len(documents)}")
    
    if not documents:
        print("No documents found!")
        return

    print(f"Loaded {len(documents)} pages. Using a subset for generation to save costs...")
    # Using specific pages to get variety (Reduced subset for stability)
    selected_docs = documents[0:50] 
    # + documents[100:105] 

    print("Initializing Ragas TestsetGenerator...")
    
    kg = get_kg(selected_docs)

    # Force use gpt-4o for dataset generation as it requires high reasoning and token limits
    # gpt-5-mini was hitting max_token limits (3k)
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Providing a fallback model if MODEL_NAME is experimental/invalid
    model_to_use = MODEL_NAME if "gpt" in MODEL_NAME else "gpt-4o"
    
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_to_use))
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    transforms = get_transforms(generator_llm)
    apply_transforms(kg, transforms)

    query_distibution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"), 0.5),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.5)
    ]

    # Use default distributions by not specifying query_distribution manually
    # The generator will use default evolutions (simple, multi_context, reasoning, etc.)
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas
    )

    # === 배치 처리 로직 시작 ===
    
    # 기존 파일이 있다면 로드하여 이어쓰기 준비
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
        # 남은 개수 계산 (배치 사이즈와 남은 개수 중 작은 값 선택)
        next_batch_size = min(BATCH_SIZE, TEST_SIZE - current_count)
        
        print(f"\n[Batch Processing] {current_count}/{TEST_SIZE} 완료. 이번에 {next_batch_size}개 생성 시도...")
        
        try:
            # 배치만큼 생성
            batch_testset = generator.generate(
                testset_size=next_batch_size,
                query_distribution=query_distibution,
            )
            
            batch_df = batch_testset.to_pandas()
            
            # 데이터프레임 병합
            if existing_df.empty:
                existing_df = batch_df
            else:
                existing_df = pd.concat([existing_df, batch_df], ignore_index=True)
            
            # 중간 저장 (덮어쓰기)
            existing_df.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=4)
            
            current_count = len(existing_df)
            print(f"--> 성공! 현재까지 총 {current_count}개 저장 완료. (파일: {OUTPUT_FILE})")
            
        except Exception as e:
            print(f"Error during batch generation: {e}")
            print("중단되었습니다. 해결 후 다시 실행하면 이어서 진행됩니다.")
            break 

    print(f"\n최종 완료. 총 {len(existing_df)}개의 데이터가 {OUTPUT_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(generate_dataset_in_batches_merged_docs())