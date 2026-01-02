import os
import json
import asyncio
import numpy as np
import pandas as pd
import time
from openai import AsyncOpenAI
from ragas.metrics.collections import ContextRelevance
from ragas.llms import llm_factory
from src.config import DATA_DIR, MODEL_NAME
from src.rag_best_practices.retrieval import get_ensemble_retriever, get_hyde_retriever, get_vectorstore

EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"
RESULTS_FILE = DATA_DIR / "experiment_results_retrieval.csv"

# Configuration for evaluation
CHUNK_SIZE = 256  # Fixed based on chunk size experiment results
ALPHA_VALUES = [0.3, 0.5, 0.7] # Alpha for Hybrid Search (0.5 = Equal weight)

async def evaluate_retriever(retriever, dataset, metric_name="ContextRelevance"):
    """
    Evaluates a specific retriever instance against the dataset using Ragas.
    """
    print(f"Evaluating {metric_name}...")
    
    scorer = ContextRelevance(
        llm=llm_factory(MODEL_NAME, client=AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))),
    )
    
    results = {
        "context_relevance": [],
        "latency": [],
        "hit_rate": [],
        "mrr": []
    }
    
    for item in dataset:
        query = item['user_input']
        # Ground Truth can be a list (reference_contexts) or string
        ground_truth_contexts = item.get('reference_contexts', []) or [item.get('ground_truth', '')]
        if isinstance(ground_truth_contexts, str):
            ground_truth_contexts = [ground_truth_contexts]
        
        # Retrieval
        try:
            start_time = time.time()
            docs = retriever.invoke(query)
            end_time = time.time()
            
            latency = end_time - start_time
            results["latency"].append(latency)
            
            retrieved_contexts = [doc.page_content for doc in docs]
            
            # 1. Hit Rate (Recall proxy) & MRR
            is_hit = False
            reciprocal_rank = 0.0
            
            for rank, doc_content in enumerate(retrieved_contexts):
                # Check if ANY of the ground truth contexts match this doc
                for gt_ctx in ground_truth_contexts:
                    if not gt_ctx: continue
                    # Check first 50 chars (cleaned) to avoid formatting mismatch
                    signature = gt_ctx.strip()[:50]
                    if signature and signature in doc_content: 
                        is_hit = True
                        reciprocal_rank = 1.0 / (rank + 1)
                        break
                if is_hit:
                    break
            
            results["hit_rate"].append(1.0 if is_hit else 0.0)
            results["mrr"].append(reciprocal_rank)

            # 2. Context Relevance (Ragas)
            ragas_result = await scorer.ascore(user_input=query, retrieved_contexts=retrieved_contexts)
            score = ragas_result.score if hasattr(ragas_result, 'score') else ragas_result.value
            results["context_relevance"].append(score)
            
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            results["context_relevance"].append(0.0)
            results["latency"].append(0.0)
            results["hit_rate"].append(0.0)
            results["mrr"].append(0.0)
            
    return results

async def main():
    if not EVAL_DATASET_PATH.exists():
        print("Evaluation dataset not found!")
        return
        
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation items.")

    all_results = []

    def process_results(method_name, results_dict):
        # Helper to compute means and append to all_results
        cr = results_dict["context_relevance"]
        lat = results_dict["latency"]
        hr = results_dict["hit_rate"]
        mrr = results_dict["mrr"]
        
        all_results.append({
            "Method": method_name,
            "Context Relevance": np.mean(cr),
            "Latency (s)": np.mean(lat),
            "Hit Rate": np.mean(hr),
            "MRR": np.mean(mrr)
        })
        print(f"{method_name} Results:")
        print(f"  CR: {np.mean(cr):.4f}, Latency: {np.mean(lat):.4f}s, HitRate: {np.mean(hr):.4f}, MRR: {np.mean(mrr):.4f}")

    # 1. Evaluate Baseline (Vector Only)
    print("\n=== Evaluating Baseline (Dense Vector Only) ===")
    vectorstore = get_vectorstore(CHUNK_SIZE)
    baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    baseline_results = await evaluate_retriever(baseline_retriever, eval_data, "Baseline")
    process_results("Baseline (Dense)", baseline_results)

    # 2. Evaluate Hybrid Search (Varying Alpha) (Reduced for speed: just check 0.5)
    # Use ALPHA_VALUES if full experiment
    for alpha in ALPHA_VALUES:
        print(f"\n=== Evaluating Hybrid Search (Alpha={alpha}) ===")
        hybrid_retriever = get_ensemble_retriever(chunk_size=CHUNK_SIZE, alpha=alpha)
        
        hybrid_results = await evaluate_retriever(hybrid_retriever, eval_data, f"Hybrid (a={alpha})")
        process_results(f"Hybrid (Alpha={alpha})", hybrid_results)

    # 3. Evaluate HyDE
    print("\n=== Evaluating HyDE ===")
    hyde_search_fn = get_hyde_retriever(chunk_size=CHUNK_SIZE)
    class HydeRetrieverWrapper:
        def invoke(self, q): return hyde_search_fn(q)
        
    hyde_results = await evaluate_retriever(HydeRetrieverWrapper(), eval_data, "HyDE")
    process_results("HyDE", hyde_results)

    # Save Results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nExperiment complete. Results saved to {RESULTS_FILE}")
    print(df)

if __name__ == "__main__":
    asyncio.run(main())
