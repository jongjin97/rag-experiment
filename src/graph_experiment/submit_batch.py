
import os
import json
import glob
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY
import sys

def submit_all_batches(batch_dir: Path, target_files: list = None):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Verify directory
    if not batch_dir.exists():
        print(f"Batch directory not found: {batch_dir}")
        return

    # Determine files to process
    if target_files:
        batch_files = []
        for fname in target_files:
            fpath = batch_dir / fname
            if fpath.exists():
                batch_files.append(str(fpath))
            else:
                print(f"Warning: Targeted file not found: {fname}")
    else:
        # Default: Process all
        file_pattern = str(batch_dir / "batch_input_part_*.jsonl")
        batch_files = glob.glob(file_pattern)
    
    if not batch_files:
        print("No batch input files found to submit.")
        return
        
    print(f"Found {len(batch_files)} files to submit.")
    
    # Load existing jobs (if any)
    record_file = batch_dir / "submitted_jobs.json"
    if record_file.exists():
        with open(record_file, "r", encoding="utf-8") as f:
            job_records = json.load(f)
    else:
        job_records = []
    
    for file_path in batch_files:
        print(f"\nUploading: {os.path.basename(file_path)}")
        
        # 1. Upload File
        try:
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
        except Exception as e:
            print(f"  - Error submitting {os.path.basename(file_path)}: {e}")

    # Save job records
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(job_records, f, indent=2)
        
    print(f"\nSubmitted jobs details saved to {record_file}")

if __name__ == "__main__":
    BATCH_DIR = Path("./data/processed_experiment/batch_input")
    
    # Check for command line arguments
    targets = None
    if len(sys.argv) > 1:
        targets = sys.argv[1:] # usage: python submit_batch.py file1.jsonl file2.jsonl
        print(f"Targeting specific files: {targets}")
        
    submit_all_batches(BATCH_DIR, target_files=targets)
