import json
import uuid
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path

from src.config import DATA_DIR, LIGHT_RAG_DIR, MODEL_NAME, SAMSUNG_DIR
from src.utils.document_loader import load_documents_merged
from src.naive_rag.ingest import split_documents
from src.prompts.prompt import GRAPH_RAG_EXTRACTOR_PROMPT
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
import tiktoken

# Safety Limit (User requested ~1.5M tokens per batch file)
TOKENS_PER_FILE_LIMIT = 1_500_000 

# Ensure LightRAG directory exists
LIGHT_RAG_DIR.mkdir(parents=True, exist_ok=True)
INDEXER_DIR = LIGHT_RAG_DIR / "indexer"
INDEXER_DIR.mkdir(parents=True, exist_ok=True)

# Data Models for Extraction Schema
class Entity(BaseModel):
    name: str = Field(description="Name of the entity, capitalized and deduped.")
    type: str = Field(description="Type of the entity (ORGANIZATION, PRODUCT, etc).")
    description: str = Field(description="Brief description of the entity.")

class Relationship(BaseModel):
    source: str = Field(description="Name of the source entity.")
    target: str = Field(description="Name of the target entity.")
    relation_type: str = Field(description="Type of relationship.")
    description: str = Field(description="Contextual explanation.")

class GraphExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


def count_tokens(text: str, model: str = "gpt-4-0125-preview") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def prepare_lightrag_batch():
    """
    1. Load Docs -> Chunk
    2. Save chunks.json (id -> text)
    3. Generate batch.jsonl files (Split by token limit)
    """
    print("Loading documents...")
    # Load Samsung reports
    docs = load_documents_merged(SAMSUNG_DIR)
    print(f"Loaded {len(docs)} documents.")
    
    # Chunking
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
    print(f"Total Chunks generated: {len(chunks)}")
    
    # Tool Schema
    tool_schema = convert_to_openai_tool(GraphExtraction)
    
    chunk_map = {}
    
    # Batch splitting logic
    current_batch_requests = []
    current_tokens = 0
    file_index = 1
    files_created = []

    print("Preparing batch requests...")
    for i, chunk in enumerate(tqdm(chunks)):
        chunk_id = f"chunk_{i}"
        
        # Save to map
        chunk_map[chunk_id] = {
            "content": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown"),
             "page": chunk.metadata.get("page", 0)
        }
        
        # Estimate tokens
        # content + system prompt + overhead
        tokens_est = count_tokens(chunk.page_content) + 1000
        
        # Construct Batch Request
        request_body = {
            "custom_id": chunk_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": GRAPH_RAG_EXTRACTOR_PROMPT},
                    {"role": "user", "content": f"Context:\n{chunk.page_content}"}
                ],
                "tools": [tool_schema],
                "tool_choice": {"type": "function", "function": {"name": "GraphExtraction"}},
                "temperature": 0.0
            }
        }
        
        # Check limit
        if current_tokens + tokens_est > TOKENS_PER_FILE_LIMIT:
             # Save current batch
             save_batch_file(current_batch_requests, file_index)
             files_created.append(file_index)
             
             # Reset
             file_index += 1
             current_batch_requests = []
             current_tokens = 0
             
        current_batch_requests.append(request_body)
        current_tokens += tokens_est

    # Save last batch
    if current_batch_requests:
        save_batch_file(current_batch_requests, file_index)
        files_created.append(file_index)

    # Save chunks.json
    chunks_file = INDEXER_DIR / "chunks.json"
    print(f"Saving chunk map to {chunks_file}...")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
            
    print(f"Preparation Complete. Created {len(files_created)} batch files.")

def save_batch_file(requests, index):
    filename = INDEXER_DIR / f"batch_input_part_{index}.jsonl"
    print(f"Writing {len(requests)} requests to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    prepare_lightrag_batch()
