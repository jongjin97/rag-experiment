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
from src.rag_best_practices.retrieval import get_ensemble_retriever, get_vectorstore
from src.rag_best_practices.reranking import BGEReranker, TILDEReranker
from langchain_classic.retrievers import EnsembleRetriever

EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"
RESULTS_FILE = DATA_DIR / "experiment_results_reranking.csv"

# Configuration
CHUNK_SIZE = 256
HYBRID_ALPHA = 0.5
RETRIEVAL_TOP_K = 20 # Retrieve 20 candidates
RERANK_TOP_K = 5     # Final selection

async def evaluate_pipeline(pipeline_fn, dataset, method_name):
    """
    Evaluates a full pipeline (Retrieval + Rerank)
    pipeline_fn: function(query) -> list of docs
    """
    print(f"\nEvaluating {method_name}...")
    
    scorer = ContextRelevance(
        llm=llm_factory(MODEL_NAME, client=AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))),
    )
    
    results = {
        "context_relevance": [],
        "latency": [],
        "hit_rate": [],
        "mrr": []
    }
    
    for i, item in enumerate(dataset):
        query = item['user_input']
        ground_truth_contexts = item.get('reference_contexts', []) or [item.get('ground_truth', '')]
        if isinstance(ground_truth_contexts, str):
            ground_truth_contexts = [ground_truth_contexts]
            
        try:
            start_time = time.time()
            # Execute Pipeline
            # Note: pipeline_fn might be async or sync, handling both if needed, but assuming sync wrapper here
            docs = pipeline_fn(query)
            end_time = time.time()
            
            latency = end_time - start_time
            results["latency"].append(latency)
            
            retrieved_contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
            
            # Hit Rate & MRR
            is_hit = False
            reciprocal_rank = 0.0
            
            for rank, doc_content in enumerate(retrieved_contexts):
                for gt_ctx in ground_truth_contexts:
                    if not gt_ctx: continue
                    signature = gt_ctx.strip()[:50]
                    if signature and signature in doc_content:
                        is_hit = True
                        reciprocal_rank = 1.0 / (rank + 1)
                        break
                if is_hit: break
            
            results["hit_rate"].append(1.0 if is_hit else 0.0)
            results["mrr"].append(reciprocal_rank)
            
            # Context Relevance
            ragas_result = await scorer.ascore(user_input=query, retrieved_contexts=retrieved_contexts)
            score = ragas_result.score if hasattr(ragas_result, 'score') else ragas_result.value
            results["context_relevance"].append(score)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} items... (Last CR: {score:.4f})", flush=True)
                
        except Exception as e:
            print(f"Error processing query '{query}': {e}", flush=True)
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
    
    # Optimization: Use subset for faster CPU experiment
    eval_data = eval_data[:20]
    print(f"Loaded {len(eval_data)} evaluation items (Subset for CPU experiment).")

    # Initialize Components
    # 1. Retrieval (Hybrid Top-20 for Reranking, Top-5 for Baseline)
    print("Initializing Retrievers...")
    hybrid_retriever_base = get_ensemble_retriever(chunk_size=CHUNK_SIZE, alpha=HYBRID_ALPHA) # default k=5
    
    # We need a retriever that returns more docs (k=20) for reranking candidates
    # Since get_ensemble_retriever creates internal retrievers with fixed k=5, we need to manually adjust or re-create
    # To keep it simple, we'll just modify the `k` of the underlying retrievers if possible, or re-instantiate.
    # Actually, EnsembleRetriever's `k` isn't strictly enforced on invoke? 
    # Let's rebuild specific components for K=20
    
    # Re-using the get_ensemble_retriever but modifying underlying retrievers is tricky.
    # Let's rely on retrieving more by configuring the vectorstore/bm25 retrievers inside.
    # Re-implementing simplified hybrid retrieval here for Top-K control:
    vectorstore = get_vectorstore(CHUNK_SIZE)
    from src.rag_best_practices.retrieval import get_bm25_retriever
    bm25_retriever = get_bm25_retriever(CHUNK_SIZE)
    bm25_retriever.k = RETRIEVAL_TOP_K
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    
    hybrid_retriever_large = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[HYBRID_ALPHA, 1.0 - HYBRID_ALPHA]
    )
    
    # 2. Rerankers
    print("Initializing Rerankers (This may take a while to download models)...")
    # Using 'base' model for CPU efficiency as discussed with user, or 'm3' if powerful enough. 
    # Plan said 'v2-m3'. User warned about speed. I'll stick to 'v2-m3' but warn.
    # Actually, to avoid 10s/query delay freezing the experiment, maybe I should switch to 'base' if possible?
    # I'll stick to 'BAAI/bge-reranker-v2-m3' as planned but note latency.
    bge_reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3") 
    tilde_reranker = TILDEReranker(model_name="ielab/TILDE")

    all_results = []
    
    def process_and_save(method, res_dict):
        cr = np.mean(res_dict["context_relevance"])
        lat = np.mean(res_dict["latency"])
        hr = np.mean(res_dict["hit_rate"])
        mrr = np.mean(res_dict["mrr"])
        
        print(f"Results for {method}: CR={cr:.4f}, Latency={lat:.4f}s, HitRate={hr:.4f}, MRR={mrr:.4f}")
        
        all_results.append({
            "Method": method,
            "Context Relevance": cr,
            "Latency (s)": lat,
            "Hit Rate": hr,
            "MRR": mrr
        })
        
    # --- Experiment ---
    
    # 1. Baseline: Hybrid Top-5 (Direct)
    # Note: re-using hybrid_retriever_base (which implicitly uses k=5 from retrieval.py defaults)
    # Or explicitly creating one with k=5
    bm25_k5 = get_bm25_retriever(CHUNK_SIZE) 
    bm25_k5.k = 5
    dense_k5 = vectorstore.as_retriever(search_kwargs={"k": 5})
    hybrid_k5 = EnsembleRetriever(retrievers=[bm25_k5, dense_k5], weights=[HYBRID_ALPHA, 1-HYBRID_ALPHA])
    
    def baseline_pipeline(q):
        return hybrid_k5.invoke(q)
        
    res_baseline = await evaluate_pipeline(baseline_pipeline, eval_data, "Baseline (Hybrid Top-5)")
    process_and_save("Baseline (Hybrid)", res_baseline)

    # 2. DLM Reranking (BGE)
    # Fetch 20 -> Rerank -> Top 5
    def bge_pipeline(q):
        # 1. Retrieve Candidates
        candidates = hybrid_retriever_large.invoke(q) 
        # 2. Extract content
        docs = [d.page_content for d in candidates]
        # 3. Rerank
        reranked_docs, _ = bge_reranker.rerank(q, docs, top_k=RERANK_TOP_K)
        # 4. Wrap back to objects (simplified, losing metadata for metric calculation but content sufficient)
        # Note: We need objects with page_content
        from langchain_core.documents import Document
        return [Document(page_content=d) for d in reranked_docs]
        
    res_bge = await evaluate_pipeline(bge_pipeline, eval_data, "DLM (BGE Reranker)")
    process_and_save("DLM (BGE)", res_bge)

    # 3. TILDE Reranking
    def tilde_pipeline(q):
        candidates = hybrid_retriever_large.invoke(q)
        docs = [d.page_content for d in candidates]
        reranked_docs, _ = tilde_reranker.rerank(q, docs, top_k=RERANK_TOP_K)
        from langchain_core.documents import Document
        return [Document(page_content=d) for d in reranked_docs]
        
    res_tilde = await evaluate_pipeline(tilde_pipeline, eval_data, "TILDE (Query Likelihood)")
    process_and_save("TILDE", res_tilde)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nReranking Experiment Complete. Saved to {RESULTS_FILE}")
    print(df)

if __name__ == "__main__":
    asyncio.run(main())
