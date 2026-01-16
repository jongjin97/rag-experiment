import json
import os
import time
from pathlib import Path
from openai import OpenAI
from src.config import LIGHT_RAG_DIR, OPENAI_API_KEY

BATCH_DIR = LIGHT_RAG_DIR / "indexer"
META_FILE = BATCH_DIR / "batch_meta.json"
OUTPUT_FILE = BATCH_DIR / "batch_output.jsonl"

def check_and_download():
    if not META_FILE.exists():
        print("No active batch metadata found.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    with open(META_FILE, "r") as f:
        batches = json.load(f)
        
    if not isinstance(batches, list):
        batches = [batches]
        
    all_completed = True
    updated_batches = []
    
    print(f"Checking status for {len(batches)} batches...")
    
    # We will append results to output file, so we don't overwrite if we run this multiple times?
    # Actually, we should check if a batch is already 'downloaded'.
    
    for batch in batches:
        batch_id = batch["batch_id"]
        status = batch.get("status")
        filename = batch.get("filename")
        
        if status == "downloaded":
             updated_batches.append(batch)
             continue
             
        try:
            job = client.batches.retrieve(batch_id)
            current_status = job.status
            print(f"[{filename}] Status: {current_status} ({job.request_counts.completed}/{job.request_counts.total})")
            
            if current_status == "completed":
                print(f"   -> Downloading results...")
                output_file_id = job.output_file_id
                if output_file_id:
                    content = client.files.content(output_file_id).text
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(content + "\n")
                    
                    batch["status"] = "downloaded"
                    print(f"   -> Saved to {OUTPUT_FILE}")
                else:
                    print("   -> Error: No output file ID.")
            elif current_status in ["failed", "cancelled", "expired"]:
                print(f"   -> Job failed or cancelled.")
                batch["status"] = current_status
                if job.errors:
                    print(f"   -> Errors: {job.errors}")
            else:
                all_completed = False
                batch["status"] = current_status
                
        except Exception as e:
            print(f"Error checking {batch_id}: {e}")
            all_completed = False
            
        updated_batches.append(batch)
        
    # Save updated meta
    with open(META_FILE, "w") as f:
        json.dump(updated_batches, f, indent=2)

    if all_completed:
        print("\nAll batches are finished/downloaded!")
        print(f"You can now run: python -m src.light_rag.indexer.process_batch")
    else:
        print("\nSome batches are still processing. Support OpenAI Batch API takes up to 24h.")
        print("Please check again later.")

if __name__ == "__main__":
    check_and_download()
