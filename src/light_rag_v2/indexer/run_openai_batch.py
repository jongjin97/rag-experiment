import os
import json
import asyncio
import time
import glob
from pathlib import Path
from openai import OpenAI
from src.config import LIGHT_RAG_DIR, OPENAI_API_KEY

BATCH_DIR = LIGHT_RAG_DIR / "indexer"
OUTPUT_FILE = BATCH_DIR / "batch_output.jsonl"
META_FILE = BATCH_DIR / "batch_meta.json"


def run_batch_lifecycle():
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 1. Find Batch Input Files
    input_files = sorted(list(BATCH_DIR.glob("batch_input_part_*.jsonl")))
    
    if not input_files:
        print("No split batch files found (batch_input_part_*.jsonl). Run prepare_batch.py first.")
        # Fallback to single file if exists and no split files
        single_file = BATCH_DIR / "batch_input.jsonl"
        if single_file.exists():
            input_files = [single_file]
        else:
            return

    print(f"Found {len(input_files)} batch input files.")
    
    # Check if we already have active jobs
    # For simplicy in this script, we'll process one by one or check metadata
    # But usually user might want to submit all. 
    # Let's clean up old meta if we are starting fresh with new files
    if META_FILE.exists():
        print("Existing meta file found. Resuming or Clearing?")
        # Logic to be robust: If users re-ran prepare, they want new batches.
        # So we should probably ignore old meta unless it matches current filename.
        # For now, let's assume we submit everything that isn't done.
    
    for input_file in input_files:
        filename = input_file.name
        print(f"\nProcessing {filename}...")
        
        # 2. Upload
        print(f"  - Uploading file...")
        batch_input_file = client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )
        file_id = batch_input_file.id
        print(f"  - File ID: {file_id}")

    # 3. Create Job
        print(f"  - Creating Batch Job...")
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "lightrag_extraction", "filename": filename}
        )
        job_id = batch_job.id
        print(f"  - Batch ID: {job_id}")
        
        # Save to meta (Append logic)
        save_batch_meta(filename, job_id, file_id)

    print(f"\nAll {len(input_files)} batches have been submitted successfully.")
    print(f"Batch IDs are saved in {META_FILE}.")
    print("You can check the status later using the OpenAI Dashboard or a separate monitoring script.")

def save_batch_meta(filename, batch_id, file_id):
    meta_data = []
    if META_FILE.exists():
        try:
            with open(META_FILE, "r") as f:
                content = json.load(f)
                if isinstance(content, list):
                    meta_data = content
                else:
                    # Migration from single dict
                    meta_data = [content]
        except:
            pass
    
    meta_data.append({
        "filename": filename,
        "batch_id": batch_id,
        "file_id": file_id,
        "status": "submitted",
        "submitted_at": str(time.time())
    })
    
    with open(META_FILE, "w") as f:
        json.dump(meta_data, f, indent=2)

if __name__ == "__main__":
    # Run
    run_batch_lifecycle()
