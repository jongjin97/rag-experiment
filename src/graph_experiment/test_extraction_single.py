
import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from src.config import MODEL_NAME, DATA_DIR
from src.graph_experiment.extractor import GraphExtraction
from src.prompts.prompt import GRAPH_RAG_REFINED_PROMPT

# Load environment variables (API Key)
load_dotenv()

# --- Configuration ---
# You can override the prompt here for testing/engineering
TEST_PROMPT = GRAPH_RAG_REFINED_PROMPT
# TEST_PROMPT = """
# 당신은 지식 그래프 추출 전문가입니다.
# ... (수정할 프롬프트 내용) ...
# """

BASE_DIR = DATA_DIR / "processed_experiment"

def list_document_folders():
    folders = [d for d in BASE_DIR.iterdir() if d.is_dir() and (d / "final_chunks.json").exists()]
    return folders

def load_chunk_data(folder_path):
    json_path = folder_path / "final_chunks.json"
    tables_path = folder_path / "extracted_tables.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    tables_data = {}
    if tables_path.exists():
        with open(tables_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
            
    return chunks, tables_data

def inject_tables(text, tables_data):
    # Regex to find [TABLE_ID]
    placeholders = re.findall(r"\[(TABLE_[a-zA-Z0-9_]+)\]", text)
    processed_text = text
    
    injected_count = 0
    for table_id in placeholders:
        if table_id in tables_data:
            table_md = tables_data[table_id].get("markdown", "")
            # Replace with markdown, adding newlines for safety
            processed_text = processed_text.replace(f"[{table_id}]", f"\n\n### Table: {table_id}\n{table_md}\n\n")
            injected_count += 1
        else:
            print(f"Warning: Table ID {table_id} not found in data.")
            
    return processed_text, injected_count

def run_extraction(text):
    print("\n--- Initializing LLM ---")
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    
    # Use structured output
    # structured_llm = llm.with_structured_output(GraphExtraction)
    # Alternatively, manual tool calling for more control/debugging (closer to batch)
    
    print("Sending request to OpenAI (Structured Output)...")
    
    structured_llm = llm.with_structured_output(GraphExtraction)
    
    messages = [
        {"role": "system", "content": TEST_PROMPT},
        {"role": "user", "content": f"Context:\n{text}"}
    ]
    
    try:
        result = structured_llm.invoke(messages)
        return result
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

def main():
    print(f"Searching for data in: {BASE_DIR}")
    folders = list_document_folders()
    
    if not folders:
        print("No processed document folders found!")
        return

    # Select Folder
    print("\n[Available Documents]")
    for i, folder in enumerate(folders):
        print(f"{i}: {folder.name}")
        
    try:
        idx_str = input("\nSelect Document Index (0-N): ")
        folder_idx = int(idx_str)
        selected_folder = folders[folder_idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    print(f"\nLoading data from: {selected_folder.name}...")
    chunks, tables = load_chunk_data(selected_folder)
    print(f"Loaded {len(chunks)} chunks.")
    
    # Select Chunk
    while True:
        try:
            chunk_idx_str = input(f"\nEnter Chunk Index (0-{len(chunks)-1}) or 'q' to quit: ")
            if chunk_idx_str.lower() == 'q':
                break
            
            chunk_idx = int(chunk_idx_str)
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                raise ValueError
                
            raw_text = chunks[chunk_idx]
            
            # Preview
            print(f"\n[Chunk {chunk_idx} Preview (First 200 chars)]")
            print("-" * 40)
            print(raw_text[:200] + "...")
            print("-" * 40)
            
            # Inject Tables
            final_text, table_count = inject_tables(raw_text, tables)
            if table_count > 0:
                print(f"-> Injected {table_count} tables into text.")
            print(f"\n Final Text: {final_text}")
            confirm = input("Run extraction on this chunk? (y/n): ")
            if confirm.lower() != 'y':
                continue
                
            # Run Extraction
            result = run_extraction(final_text)
            
            if result:
                print("\n" + "="*50)
                print("EXTRACTION RESULT")
                print("="*50)
                
                print(f"\nEntities ({len(result.entities)}):")
                for e in result.entities:
                    print(f" - [{e.type}] {e.name}: {e.description}")
                    
                print(f"\nRelationships ({len(result.relationships)}):")
                for r in result.relationships:
                    print(f" - {r.source} -> {r.relation_type} -> {r.target} ({r.description})")
            
        except ValueError:
            print("Invalid input.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
