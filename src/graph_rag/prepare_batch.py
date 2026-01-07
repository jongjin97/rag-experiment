import json
import os
import tiktoken
from pathlib import Path
from tqdm import tqdm
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.utils.document_loader import load_documents_merged
from src.rag_best_practices.chunking import split_documents
from src.config import DATA_DIR, MODEL_NAME
from src.graph_rag.extractor import GraphExtractor, GraphExtraction
from src.prompts.prompt import GRAPH_RAG_EXTRACTOR_PROMPT

# Output Directory
BATCH_DIR = DATA_DIR / "graph_rag" / "batch_jobs"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
DOC_PATH = DATA_DIR / "samsung" 
# Safety Limit (User requested ~1.5M tokens per batch file)
TOKENS_PER_FILE_LIMIT = 1_500_000 

def count_tokens(text: str, model: str = "gpt-4-0125-preview") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def prepare_batch_files():
    print("Loading documents (Merged Mode)...")
    # Load all PDFs merging pages
    docs = load_documents_merged(str(DOC_PATH)) 
    print(f"Loaded {len(docs)} documents.")
    
    # Split into chunks (matching builder.py logic)
    chunks = split_documents(docs, chunk_size=1024, chunk_overlap=100)
    print(f"Total Chunks: {len(chunks)}")
    
    # Prepare Tools Schema (for Structured Output)
    # Using LangChain utils to get exact OpenAI tool definition from Pydantic model
    tool_schema = convert_to_openai_tool(GraphExtraction)
    
    files_created = []
    current_tokens = 0
    current_requests = []
    file_index = 1
    
    for i, chunk in enumerate(tqdm(chunks, desc="Preparing Requests")):
        text_content = chunk.page_content
        
        # Estimate Input Tokens (System + User + Tool Def)
        # Detailed accounting isn't 100% precise but good estimate
        # content tokens + approx 500 overhead for system/tools
        tokens_est = count_tokens(text_content) + 500 
        
        # Construct Request Body
        # Must match what ChatOpenAI.with_structured_output sends
        request_body = {
            "custom_id": f"chunk_{i}", # ID to map back result
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME, # gpt-4.1-nano (or whatever is in config)
                "messages": [
                    {"role": "system", "content": GRAPH_RAG_EXTRACTOR_PROMPT},
                    {"role": "user", "content": f"Context:\n{text_content}"}
                ],
                "tools": [tool_schema],
                "tool_choice": {"type": "function", "function": {"name": tool_schema["function"]["name"]}},
                "temperature": 0.0,
                "max_tokens": 2000
            }
        }
        
        # Check Limit
        if current_tokens + tokens_est > TOKENS_PER_FILE_LIMIT:
            # Save current batch
            save_batch_file(current_requests, file_index)
            files_created.append(file_index)
            file_index += 1
            current_requests = []
            current_tokens = 0
            
        current_requests.append(request_body)
        current_tokens += tokens_est
        
    # Save last batch
    if current_requests:
        save_batch_file(current_requests, file_index)
        files_created.append(file_index)
        
    print(f"\nSuccessfully created {len(files_created)} batch files in {BATCH_DIR}")

def save_batch_file(requests, index):
    filename = BATCH_DIR / f"merged_batch_requests_part_{index}.jsonl"
    print(f"Writing {len(requests)} requests to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    prepare_batch_files()
