import os
import asyncio
import openai
from pathlib import Path
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.persona import Persona
from src.utils.document_loader import load_documents
from src.config import DATA_DIR, MODEL_NAME
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI

# Configuration
TEST_SIZE = 25
OUTPUT_FILE = DATA_DIR / "eval_dataset.json"
DOCUMENT_DIR = DATA_DIR / "samsung" / "[삼성전자] 반기보고서(일반법인) (2025.08.14).pdf"

# Define Personas for Samsung Report Analysis
personas = [
    Persona(
        name="Financial Analyst",
        role_description="A senior financial analyst analyzing Samsung Electronics' quarterly and annual reports. Focuses on revenue, operating profit, debt ratio, and financial stability.",
    ),
    Persona(
        name="Tech Investor",
        role_description="An investor looking for growth potential in Samsung's DX (Device eXperience) and DS (Device Solutions) divisions. Interested in market share, new product launches, and R&D investments.",
    ),
    Persona(
        name="Risk Manager",
        role_description="A risk assessment specialist identifying potential business risks, regulatory issues, and supply chain challenges mentioned in the business report.",
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
    headline_extractor = HeadlinesExtractor(llm=llm, max_num=20)
    headline_splitter = HeadlineSplitter(max_tokens=1500)
    keyphrase_extractor = KeyphrasesExtractor(llm=llm)
    transforms = [headline_extractor, headline_splitter, keyphrase_extractor]
    return transforms

async def generate_dataset():
    print("Loading documents...")
    documents = load_documents(DOCUMENT_DIR)
    
    if not documents:
        print("No documents found!")
        return

    print(f"Loaded {len(documents)} pages. Using a subset for generation to save costs...")
    # Using specific pages to get variety (Reduced subset for stability)
    selected_docs = documents[:10] + documents[100:110] 

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

    print(f"Generating {TEST_SIZE} test samples...")
    
    testset = generator.generate(
        testset_size=TEST_SIZE,
        query_distribution=query_distibution,
    )

    print("Saving dataset...")
    # Convert to pandas then JSON
    df = testset.to_pandas()
    print(df)
    # Save as JSON records
    df.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=4)
    print(f"Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(generate_dataset())
