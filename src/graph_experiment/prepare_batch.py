
import json
import os
import re
import tiktoken
from pathlib import Path
from tqdm import tqdm
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.config import MODEL_NAME
from src.prompts.prompt import GRAPH_RAG_HYBRID_PROMPT
from src.graph_experiment.extractor import GraphExtraction

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def prepare_batch_files(
    data_dir: Path, 
    output_dir: Path, 
    max_tokens_per_file: int = 1500000
):
    """
    Reads all final_chunks.json, creates batch requests using OpenAI Function Calling (Tools),
    and splits them into .jsonl files.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # 1. Prepare Tool Schema
    # This aligns with graph_rag_v2/prepare_batch.py using convert_to_openai_tool
    tool_schema = convert_to_openai_tool(GraphExtraction)
    
    # 2. Collect all chunks
    all_chunks = []
    
    # Iterate document folders
    dirs_to_process = [
        d for d in data_dir.iterdir() 
        if d.is_dir() and (d / "final_chunks.json").exists()
    ]
    
    print(f"Found {len(dirs_to_process)} document folders.")
    
    for doc_dir in dirs_to_process:
        json_path = doc_dir / "final_chunks.json"
        tables_path = doc_dir / "extracted_tables.json"
        
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        # Load tables data
        tables_data = {}
        if tables_path.exists():
            with open(tables_path, "r", encoding="utf-8") as f:
                tables_data = json.load(f)
            print(f"  - Loaded {len(tables_data)} tables from {tables_path.name}")
        
        print(f"  - {doc_dir.name}: {len(chunks)} chunks")
        
        # Add metadata to track origin and Replace Placeholders
        for idx, chunk_text in enumerate(chunks):
            # Replace placeholders: [TABLE_ID] -> markdown
            placeholders = re.findall(r"\[(TABLE_[a-zA-Z0-9_]+)\]", chunk_text)
            
            processed_text = chunk_text
            for table_id in placeholders:
                if table_id in tables_data:
                    table_md = tables_data[table_id].get("markdown", "")
                    processed_text = processed_text.replace(f"[{table_id}]", f"\n{table_md}\n")
                else:
                    print(f"    ! Warning: Table ID {table_id} not found in {doc_dir.name}")

            # Custom ID Format: DOCNAME_CHUNKINDEX
            safe_doc_name = doc_dir.name.replace(" ", "_").replace("[", "").replace("]", "")
            custom_id = f"{safe_doc_name}_{idx}"
            
            all_chunks.append({
                "custom_id": custom_id,
                "text": processed_text
            })

    print(f"Total Chunks to Process: {len(all_chunks)}")
    
    # 3. Create Batches
    current_batch = []
    current_tokens = 0
    file_counter = 1
    
    for item in tqdm(all_chunks, desc="Creating Batch Requests"):
        text_content = item["text"]
        
        # Construct Request Body using Tools
        request_body = {
            "custom_id": item["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": GRAPH_RAG_HYBRID_PROMPT},
                    {"role": "user", "content": f"Context:\n{text_content}"}
                ],
                "tools": [tool_schema],
                "tool_choice": {"type": "function", "function": {"name": tool_schema["function"]["name"]}},
                "temperature": 0.0,
            }
        }
        
        # Estimate token size (content + overhead)
        # Overhead for tools definition approx 500-1000 tokens depending on schema complexity
        req_tokens = count_tokens(text_content) + 3000 
        
        if current_tokens + req_tokens > max_tokens_per_file:
            save_batch_file(current_batch, output_dir, file_counter)
            file_counter += 1
            current_batch = []
            current_tokens = 0
            
        current_batch.append(request_body)
        current_tokens += req_tokens
        
    # Save last batch
    if current_batch:
        save_batch_file(current_batch, output_dir, file_counter)

def save_batch_file(batch_data: list, output_dir: Path, file_num: int):
    filename = output_dir / f"batch_input_part_{file_num}.jsonl"
    
    with open(filename, "w", encoding="utf-8") as f:
        for req in batch_data:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            
    print(f"Saved: {filename} ({len(batch_data)} requests)")

if __name__ == "__main__":
    DATA_DIR = Path("./data/processed_experiment")
    BATCH_DIR = DATA_DIR / "batch_input"
    
    prepare_batch_files(DATA_DIR, BATCH_DIR)
