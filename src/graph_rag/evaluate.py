import json
import time
import pandas as pd
from tqdm import tqdm
import os
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.graph_rag.retriever import GraphRetriever
from src.config import DATA_DIR, MODEL_NAME, HUGGINGFACE_EMBEDDING_MODEL

EVAL_DIR = DATA_DIR / "evaluation"
TESTSET_FILE = DATA_DIR / "eval_dataset_merged.json"
RESULTS_FILE = DATA_DIR / "graph_rag" /"benchmark_results_graph_2.csv"

def load_testset():
    if not TESTSET_FILE.exists():
        raise FileNotFoundError(f"Testset not found at {TESTSET_FILE}")
        
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {TESTSET_FILE}")
    return data[:100]

def run_inference_and_evaluate():
    print("Loading Testset...")
    raw_data = load_testset()
    
    # Map fields from eval_dataset_merged.json
    # user_input -> question
    # reference -> ground_truth (for reference, though faithfulness uses ctx/answer)
    # persona_name -> category
    questions = [d.get("user_input") for d in raw_data]
    ground_truths = [d.get("reference") for d in raw_data]
    
    # Handle persona_name or use default
    categories = [d.get("persona_name", "General") for d in raw_data]
    
    print("Initializing GraphRetriever (v2)...")
    graph_retriever = GraphRetriever()
    
    # Initialize Eval LLM & Embeddings (for RAGAS)
    print("Initializing Evaluator Models...")
    eval_llm = ChatOpenAI(model=MODEL_NAME)
    eval_embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)

    # Mode: Hybrid Only
    mode_name = "Graph Hybrid"
    
    print(f"\n--- Evaluating Mode: {mode_name} ---")
    
    mode_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "category": [],
        "latency": []
    }
    
    # 1. Inference Phase
    print(f"Starting Inference for {len(questions)} questions...")
    for idx, q in enumerate(tqdm(questions, desc=f"Inference ({mode_name})")):
        start_time = time.time()
        try:
            # GraphRAG v2 Hybrid Query
            response = graph_retriever.query(q, mode="hybrid")
            
            # Response handling
            if isinstance(response, dict):
                ans = response.get("result", str(response))
                # context is a list of strings
                ctx = response.get("context", [])
            else:
                ans = str(response)
                ctx = ["No context captured"]
                
        except Exception as e:
            print(f"Error in {mode_name} for q='{q[:20]}...': {e}")
            ans = "Error"
            ctx = []
        
        elapsed = time.time() - start_time
        
        mode_data["question"].append(q)
        mode_data["answer"].append(ans)
        mode_data["contexts"].append(ctx)
        mode_data["ground_truth"].append(ground_truths[idx])
        mode_data["category"].append(categories[idx])
        mode_data["latency"].append(elapsed)

    # 2. Evaluation Phase (RAGAS)
    print("Preparing RAGAS Dataset...")
    dataset_dict = {
        "question": mode_data["question"],
        "answer": mode_data["answer"],
        "contexts": mode_data["contexts"],
        "ground_truth": mode_data["ground_truth"]
    }
    ds = Dataset.from_dict(dataset_dict)
    
    print(f"Running RAGAS Metrics for {mode_name}...")
    try:
        # Evaluate using Faithfulness and Answer Relevancy
        ragas_results = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False 
        )
        ragas_df = ragas_results.to_pandas()
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        # Return empty DF with expected columns to prevent crash in merge
        ragas_df = pd.DataFrame(columns=["faithfulness", "answer_relevancy"])
    
    # 3. Merge Results and Save
    all_results = []
    for i in range(len(mode_data["question"])):
        # Safely get metrics
        faith_score = 0.0
        rel_score = 0.0
        
        if i < len(ragas_df):
            faith_score = ragas_df.iloc[i].get("faithfulness", 0.0)
            rel_score = ragas_df.iloc[i].get("answer_relevancy", 0.0)

        row = {
            "Mode": mode_name,
            "Category": mode_data["category"][i],
            "Question": mode_data["question"][i],
            "Answer": mode_data["answer"][i],
            "Ground Truth": mode_data["ground_truth"][i],
            "Latency": round(mode_data["latency"][i], 2),
            "Faithfulness": faith_score,
            "Answer Relevance": rel_score,
        }
        all_results.append(row)

    # Save to CSV
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    print(f"\nFinal Benchmark Results with RAGAS Saved to {RESULTS_FILE}")
    
    # Summary
    print("\n--- Benchmark Summary (Averages) ---")
    summary = final_df.groupby("Mode")[["Latency", "Faithfulness", "Answer Relevance"]].mean()
    print(summary)
    
    # Save Summary Report
    summary.to_markdown(EVAL_DIR / "summary_report_hybrid_2.md")

if __name__ == "__main__":
    run_inference_and_evaluate()
