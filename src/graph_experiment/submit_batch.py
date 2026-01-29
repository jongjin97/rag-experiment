
import os
import json
import glob
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY
def submit_all_batches(batch_dir: Path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Verify directory
    if not batch_dir.exists():
        print(f"Batch directory not found: {batch_dir}")
        return

    # Find all .jsonl files
    file_pattern = str(batch_dir / "batch_input_part_*.jsonl")
    batch_files = glob.glob(file_pattern)
    
    if not batch_files:
        print("No batch input files found.")
        return
        
    print(f"Found {len(batch_files)} batch input files.")
    
    # Store job info
    job_records = []
    
    for file_path in batch_files:
        print(f"\nUploading: {file_path}")
        
        # 1. Upload File
        with open(file_path, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
            
        print(f"  - File ID: {batch_input_file.id}")
        
        # 2. Create Batch Job
        print(f"  - Submitting Batch Job...")
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h", # Currently only 24h is supported
            metadata={
                "description": "GraphRAG Experiment Batch",
                "original_file": os.path.basename(file_path)
            }
        )
        
        print(f"  - Job ID: {batch_job.id} (Status: {batch_job.status})")
        
        job_records.append({
            "file": os.path.basename(file_path),
            "file_id": batch_input_file.id,
            "job_id": batch_job.id,
            "status": batch_job.status,
            "created_at": batch_job.created_at
        })

    # Save job records
    record_file = batch_dir / "submitted_jobs.json"
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(job_records, f, indent=2)
        
    print(f"\nSubmitted all jobs. details saved to {record_file}")

if __name__ == "__main__":
    BATCH_DIR = Path("./data/processed_experiment/batch_input")
    submit_all_batches(BATCH_DIR)
