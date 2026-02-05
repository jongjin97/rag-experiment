import json
import re
from pathlib import Path
from tqdm import tqdm
from src.config import DATA_DIR, LIGHT_RAG_V2_DIR

def import_graph_experiment_data():
    """
    Migrates data from `src/graph_experiment` (processed_experiment) to `src/light_rag_v2`.
    
    1. Reconstructs `chunks.json` (ID -> Text w/ Tables)
    2. Merges `batch_input_part_X.jsonl` + results into `batch_output.jsonl`
       (Actually, we need the *OUTPUT* from the experiment. 
        Assuming `batch_output` files exist in processed_experiment or similar.)
       
    *Adjustment*: The user's prompt implies we want to reuse the *Processing Logic* of LightRAG 
    on the *Existing Data* of Graph Experiment.
    
    The Graph Experiment data is in `data/processed_experiment`.
    It has:
    - [DocName]/final_chunks.json
    - [DocName]/extracted_tables.json
    - batch_input/batch_input_part_*.jsonl
    - batch_output/batch_output_*.jsonl (Assuming these were downloaded)
    
    Target:
    - `src/light_rag_v2/indexer/chunks.json`
    - `src/light_rag_v2/indexer/batch_output.jsonl`
    """
    
    EXPERIMENT_DIR = DATA_DIR / "processed_experiment"
    TARGET_INDEXER_DIR = LIGHT_RAG_V2_DIR / "indexer"
    TARGET_INDEXER_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Migrating data from {EXPERIMENT_DIR} to {TARGET_INDEXER_DIR}...")
    
    # 1. Reconstruct Chunks (chunks.json)
    chunk_map = {}
    
    doc_dirs = [d for d in EXPERIMENT_DIR.iterdir() if d.is_dir() and (d / "final_chunks.json").exists()]
    print(f"Found {len(doc_dirs)} document directories.")
    
    for doc_dir in tqdm(doc_dirs, desc="Processing Documents"):
        # Load tables
        tables = {}
        if (doc_dir / "extracted_tables.json").exists():
            with open(doc_dir / "extracted_tables.json", "r", encoding="utf-8") as f:
                tables = json.load(f)
        
        # Load chunks
        with open(doc_dir / "final_chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        # Reconstruct text
        safe_doc_name = doc_dir.name.replace(" ", "_").replace("[", "").replace("]", "")
        
        for idx, chunk_text in enumerate(chunks):
            # Same logic as prepare_batch.py in graph_experiment
            # Replace placeholders: [TABLE_ID] -> markdown
            placeholders = re.findall(r"\[(TABLE_[a-zA-Z0-9_]+)\]", chunk_text)
            
            processed_text = chunk_text
            for table_id in placeholders:
                if table_id in tables:
                    table_md = tables[table_id].get("markdown", "")
                    processed_text = processed_text.replace(f"[{table_id}]", f"\n{table_md}\n")
            
            custom_id = f"{safe_doc_name}_{idx}"
            
            # LightRAG expects: { 'content': ..., 'source': ..., 'page': ... }
            # Graph Experiment didn't explicit save page numbers in final_chunks.json, 
            # but we can assume sequential or just leave page as 0 for now.
            chunk_map[custom_id] = {
                "content": processed_text,
                "source": doc_dir.name,
                "page": 0 # Placeholder
            }
            
    # Save chunks.json
    chunks_file = TARGET_INDEXER_DIR / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunk_map)} chunks to {chunks_file}")

    # 2. Merge Batch Outputs
    # We look for batch_output/ directory in EXPERIMENT_DIR
    # Found in `batch_input` based on investigation
    BATCH_OUTPUT_DIR = EXPERIMENT_DIR / "batch_input2"
    TARGET_BATCH_FILE = TARGET_INDEXER_DIR / "batch_output.jsonl"
    
    if not BATCH_OUTPUT_DIR.exists():
        print(f"Warning: {BATCH_OUTPUT_DIR} does not exist. Cannot migrate batch results.")
        print("Please ensure you have downloaded batch results to `data/processed_experiment/batch_output`.")
        return

    merged_lines = 0
    with open(TARGET_BATCH_FILE, "w", encoding="utf-8") as outfile:
        for jsonl_file in BATCH_OUTPUT_DIR.glob("batch_output_*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
                    merged_lines += 1
                    
    print(f"Merged {merged_lines} batch result lines to {TARGET_BATCH_FILE}")

if __name__ == "__main__":
    import_graph_experiment_data()
