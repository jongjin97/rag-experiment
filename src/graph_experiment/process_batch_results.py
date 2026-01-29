
import json
import os
from pathlib import Path
from openai import OpenAI
import src.config as config
def process_batch_results(batch_dir: Path):
    client = OpenAI()
    
    # Load submitted jobs record
    record_file = batch_dir / "submitted_jobs.json"
    if not record_file.exists():
        print("No submitted jobs record found.")
        return
        
    with open(record_file, "r") as f:
        jobs = json.load(f)
        
    print(f"Checking status for {len(jobs)} jobs...")
    
    completed_results = []
    
    for job_info in jobs:
        job_id = job_info["job_id"]
        
        # Check status
        batch_job = client.batches.retrieve(job_id)
        print(f"Job {job_id} Status: {batch_job.status}")
        
        if batch_job.status == "completed" and batch_job.output_file_id:
            # Download Results
            print(f"  - Downloading results ({batch_job.output_file_id})...")
            
            result_content = client.files.content(batch_job.output_file_id).content
            
            # Save Raw Output
            output_filename = f"batch_output_{job_id}.jsonl"
            output_path = batch_dir / output_filename
            
            with open(output_path, "wb") as f:
                f.write(result_content)
                
            print(f"  - Saved raw output to: {output_path}")
            
            # Parse and save structured data (Placeholder for now)
            # You would iterate line by line, match custom_id to doc/chunk, and save entities.
            
        elif batch_job.status == "failed" or batch_job.status == "expired" or batch_job.status == "cancelled":
            print(f"  - Job Failed/Expired: {batch_job.errors}")
            retry_failed_job(client, job_info, batch_dir, jobs)

    # Save updated job records if any retries added
    record_file = batch_dir / "submitted_jobs.json"
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

def retry_failed_job(client, job_info, batch_dir, jobs_list):
    """
    Splits the failed job's input file into two and resubmits.
    """
    original_file_id = job_info["file_id"]
    original_filename = job_info["file"]
    
    print(f"  > Initiating Retry for {original_filename} (Split & Retry)...")
    
    # 1. Download original input file content
    try:
        content = client.files.content(original_file_id).content.decode("utf-8")
    except Exception as e:
        print(f"    ! Error downloading input file: {e}")
        return

    lines = [line for line in content.splitlines() if line.strip()]
    total_lines = len(lines)
    
    if total_lines < 2:
        print("    ! Cannot split file with less than 2 requests. Skipping.")
        return
        
    mid_point = total_lines // 2
    part_a = lines[:mid_point]
    part_b = lines[mid_point:]
    
    print(f"    > Splitting {total_lines} requests into {len(part_a)} and {len(part_b)}")
    
    # 2. Create new files
    # Determine new filenames (e.g., batch_input_part_1_retry_1.jsonl)
    base_name = os.path.splitext(original_filename)[0]
    
    parts = [("partA", part_a), ("partB", part_b)]
    
    for suffix, data_lines in parts:
        new_filename = f"{base_name}_retry_{suffix}.jsonl"
        new_file_path = batch_dir / new_filename
        
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data_lines))
            
        print(f"    > Created {new_filename}")
        
        # 3. Submit new batch
        with open(new_file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
            
        new_batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Retry of {original_filename}", "original_file": new_filename}
        )
        
        print(f"    > Submitted Retry Job {new_batch_job.id}")
        
        # Add to jobs list
        jobs_list.append({
            "file": new_filename,
            "file_id": batch_input_file.id,
            "job_id": new_batch_job.id,
            "status": new_batch_job.status,
            "created_at": new_batch_job.created_at,
            "is_retry": True,
            "parent_job_id": job_info["job_id"]
        })
    
    # Mark original as retried
    job_info["status"] = "failed_retried"
            
if __name__ == "__main__":
    BATCH_DIR = Path("./data/processed_experiment/batch_input")
    process_batch_results(BATCH_DIR)
