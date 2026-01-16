import asyncio
import json
import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.light_rag.utils.embedding import get_embedding_function
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.light_rag.lightrag import LightRAG
from src.config import DATA_DIR, LIGHT_RAG_DIR, MODEL_NAME

def load_evaluation_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

async def generate_rag_responses(rag, eval_data, limit=None):
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    # If limit is set, slice the data
    target_data = eval_data[:limit] if limit else eval_data

    print(f"Generating responses for {len(target_data)} samples...")

    for i, item in enumerate(target_data):
        question = item["question"]
        
        # LightRAG query (Hybrid mode default)
        print(f"Processing {i+1}/{len(target_data)}: {question[:30]}...")
        try:
            # Get Answer
            answer = await rag.query(question, mode="hybrid")
            
            # Get Contexts (LightRAG doesn't return context explicitly in query(), 
            # so we fetch it separately for evaluation purposes using the retriever directly)
            # Note: Ideally LightRAG.query should return the context used. 
            # For now, we will simulate the retrieval step to capture context.
            retrieved_context_str = await rag.retriever.retrieve_hybrid(question)
            
            # RAGAS expects 'contexts' as a list of strings. 
            # Since retrieve_hybrid returns a single formatted string, we might wrap it in a list 
            # or try to split it if it has clear delimiters. 
            # For now, wrapping the whole context string in a list is the safest approach to pass context.
            contexts = [retrieved_context_str]

            results["question"].append(question)
            results["answer"].append(answer)
            results["contexts"].append(contexts)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    return results


def run_evaluation(results_dict):
    print("\nStarting RAGAS evaluation...")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(results_dict)
    
    # Define metrics to use
    metrics = [
        faithfulness,
        answer_relevancy,
        # context_precision, # Requires ground_truth contexts (which we might have in json but need aligning)
        # context_recall,    # Requires ground_truth contexts
    ]

    # Initialize Embeddings
    embedding_model = get_embedding_function()
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    # Initialize Models
    openai_llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
    
    # Ragas LLM Wrapper
    ragas_llm = LangchainLLMWrapper(openai_llm)

    # Run evaluation
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=ragas_embeddings,
        llm=ragas_llm
    )
    
    return scores

async def main():
    Dataset_Path = os.path.join(DATA_DIR / "evaluation", "qa_testset.json")
    
    if not os.path.exists(Dataset_Path):
        # Fallback to absolute path if running from root without relative assumption setup or if file moved
        # Trying the user specified path directly if above fails
        Dataset_Path = "data/evaluation/qa_testset.json" 
        if not os.path.exists(Dataset_Path):
            print(f"Error: Dataset not found at {Dataset_Path}")
            return

    print("Initializing LightRAG...")
    rag = LightRAG()
    
    print(f"Loading dataset from {Dataset_Path}...")
    eval_data = load_evaluation_dataset(Dataset_Path)
    
    # Generate answers and contexts
    # limit=5 for testing, remove limit=None for full run
    results = await generate_rag_responses(rag, eval_data) 
    
    # Run Evaluation
    scores = run_evaluation(results)
    
    print("\n=== Evaluation Results ===")
    print(scores)
    
    # Save results
    output_df = scores.to_pandas()
    output_path = os.path.join(DATA_DIR / "evaluation", "benchmark_results_ragas_light.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    # Windows asyncio policy fix if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
