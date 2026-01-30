
import json
from openai import OpenAI
from src.config import OPENAI_API_KEY

def inspect_batch():
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Use one of the completed job IDs from the previous output
    # batch_697b5a92e01881909d8ce04f0746be66 (batch_input_part_2.jsonl)
    job_id = "batch_697b5a92e01881909d8ce04f0746be66" 
    
    try:
        batch_job = client.batches.retrieve(job_id)
        print(batch_job.model_dump_json(indent=2))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    inspect_batch()
