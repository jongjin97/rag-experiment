import json
import os
import glob
from pathlib import Path
from openai import OpenAI
from src.config import DATA_DIR

BATCH_DIR = DATA_DIR / "graph_rag_v2" / "batch_jobs"
STATUS_FILE = BATCH_DIR / "active_batches.json"

def submit_batches():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Find generated batch files (both extraction and community summarization)
    files = sorted(list(BATCH_DIR.glob("merged_batch_requests_part_*.jsonl")) + 
                   list(BATCH_DIR.glob("community_batch_part_*.jsonl")))
    
    if not files:
        print("No batch files found. Run prepare_batch.py first.")
        return

    print(f"Found {len(files)} batch files to submit.")
    
    active_batches = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            active_batches = json.load(f)

    for file_path in files:
        filename = file_path.name
        
        # Check if already submitted
        if filename in active_batches:
            status = active_batches[filename].get('status', 'unknown')
            print(f"Skipping {filename} (Already tracked: {status}, ID: {active_batches[filename]['batch_id']})")
            continue
            
        print(f"\nUploading {filename}...")
        try:
            # 1. Upload File
            batch_input_file = client.files.create(
                file=open(file_path, "rb"),
                purpose="batch"
            )
            file_id = batch_input_file.id
            print(f"  - Uploaded File ID: {file_id}")
            
            # 2. Create Batch Job
            print(f"  - Creating Batch Job...")
            batch_job = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "graph_rag_extraction", "filename": filename}
            )
            batch_id = batch_job.id
            print(f"  - Batch Job Created! ID: {batch_id}")
            
            # 3. Save Info
            active_batches[filename] = {
                "batch_id": batch_id,
                "file_id": file_id,
                "status": "in_progress",
                "submitted_at": str(batch_job.created_at)
            }
            
        except Exception as e:
            print(f"  - Error submitting {filename}: {e}")

    # Save Tracking Info
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(active_batches, f, indent=2)
    
    print(f"\nAll jobs submitted. Tracking info saved to {STATUS_FILE}")
    print("Run this script again or check OpenAI Dashboard to monitor status.")

if __name__ == "__main__":
    submit_batches()
