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

from src.naive_rag.chain import rag_chain
from src.graph_rag_v2.retriever import GraphRetriever
from src.config import DATA_DIR, MODEL_NAME

EVAL_DIR = DATA_DIR / "evaluation"
TESTSET_FILE = EVAL_DIR / "qa_testset.json"
RESULTS_FILE = DATA_DIR / "graph_rag_v2" / "benchmark_results_graph_v2.csv"

def load_testset():
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def run_inference_and_evaluate():
    print("Loading Testset...")
    questions_data = load_testset()
    questions = [q["question"] for q in questions_data]
    categories = [q["category"] for q in questions_data]
    
    print("Initializing GraphRetriever...")
    graph_retriever = GraphRetriever()
    
    # Initialize Eval LLM & Embeddings (for RAGAS)
    print("Initializing Evaluator Models...")
    eval_llm = ChatOpenAI(model=MODEL_NAME)
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Modes to evaluate
    modes = [
        ("Naive RAG", lambda q: rag_chain(q)),
        ("Graph Local", lambda q: graph_retriever.query(q, mode="local")),
        ("Graph Global", lambda q: graph_retriever.query(q, mode="global")),
        ("Graph Hybrid", lambda q: graph_retriever.query(q, mode="hybrid")),
    ]
    
    all_results = []
    
    for mode_name, func in modes:
        print(f"\n--- Evaluating Mode: {mode_name} ---")
        
        mode_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [], # Optional, leave empty
            "category": [],
            "latency": []
        }
        
        # 1. Inference Phase
        for idx, q in enumerate(tqdm(questions, desc=f"Inference ({mode_name})")):
            start_time = time.time()
            try:
                response = func(q)
                # response check
                if isinstance(response, dict):
                    ans = response.get("result", str(response))
                    ctx = response.get("context", [])
                else:
                    # Fallback if refactoring failed or old code
                    ans = str(response)
                    ctx = ["No context captured"]
                    
            except Exception as e:
                print(f"Error in {mode_name}: {e}")
                ans = "Error"
                ctx = []
            
            elapsed = time.time() - start_time
            
            mode_data["question"].append(q)
            mode_data["answer"].append(ans)
            mode_data["contexts"].append(ctx)
            mode_data["ground_truth"].append("nan") # Placeholder
            mode_data["category"].append(categories[idx])
            mode_data["latency"].append(elapsed)

        # 2. Evaluation Phase (RAGAS)
        # Convert to Dataset
        dataset_dict = {
            "question": mode_data["question"],
            "answer": mode_data["answer"],
            "contexts": mode_data["contexts"],
            # "ground_truth": mode_data["ground_truth"] # Ragas might warn if missing, but for these metrics it's ok
        }
        ds = Dataset.from_dict(dataset_dict)
        
        print(f"Running RAGAS Metrics for {mode_name}...")
        try:
            ragas_results = evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy],
                llm=eval_llm,
                embeddings=eval_embeddings,
                raise_exceptions=False 
            )
            ragas_df = ragas_results.to_pandas()
        except Exception as e:
            print(f"RAGAS evaluation failed for {mode_name}: {e}")
            # Create empty DF with NaNs
            ragas_df = pd.DataFrame(columns=["faithfulness", "answer_relevancy"])
        
        # 3. Merge Results
        for i in range(len(mode_data["question"])):
            row = {
                "Mode": mode_name,
                "Category": mode_data["category"][i],
                "Question": mode_data["question"][i],
                "Answer": mode_data["answer"][i],
                "Latency": round(mode_data["latency"][i], 2),
                "Faithfulness": ragas_df.iloc[i].get("faithfulness", 0.0) if i < len(ragas_df) else 0.0,
                "Answer Relevance": ragas_df.iloc[i].get("answer_relevancy", 0.0) if i < len(ragas_df) else 0.0,
            }
            all_results.append(row)

    # Save to CSV
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    print(f"\nFinal Benchmark Results with RAGAS Saved to {RESULTS_FILE}")
    
    # Summary
    print("\n--- Benchmark Summary (Averages) ---")
    summary = final_df.groupby("Mode")[["Latency", "Faithfulness", "Answer Relevance"]].mean()
    print(summary)
    
    # Save Summary as Markdown for Report
    summary.to_markdown(EVAL_DIR / "summary_report.md")

if __name__ == "__main__":
    run_inference_and_evaluate()
