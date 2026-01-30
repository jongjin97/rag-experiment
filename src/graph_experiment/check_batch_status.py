
import json
import os
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY

def check_batch_status(batch_dir: Path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    record_file = batch_dir / "submitted_jobs.json"
    if not record_file.exists():
        print("No submitted jobs record found.")
        return
        
    with open(record_file, "r") as f:
        jobs = json.load(f)
        
    print(f"\nChecking status for {len(jobs)} jobs in {batch_dir.name}...\n")
    print(f"{'Job ID':<35} | {'Original File':<40} | {'Status':<15}")
    print("-" * 100)
    
    counts = {"completed": 0, "failed": 0, "in_progress": 0, "finalizing": 0, "cancelled": 0, "expired": 0}
    
    for job_info in jobs:
        job_id = job_info["job_id"]
        file_name = job_info.get("file", "unknown")
        
        try:
            batch_job = client.batches.retrieve(job_id)
            status = batch_job.status
            
            print(f"{job_id:<35} | {file_name:<40} | {status:<15}")
            
            # Update counts (simplify status)
            if status in counts:
                counts[status] += 1
            else:
                counts["in_progress"] += 1 # bucket others like 'validating'
                
        except Exception as e:
            print(f"{job_id:<35} | {file_name:<40} | Error: {e}")

    print("\n--- Summary ---")
    for k, v in counts.items():
        if v > 0:
            print(f"{k.capitalize()}: {v}")

if __name__ == "__main__":
    BATCH_DIR = Path("./data/processed_experiment/batch_input")
    check_batch_status(BATCH_DIR)
