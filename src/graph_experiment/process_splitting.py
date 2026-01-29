
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.graph_experiment.text_splitter import TableAwareSplitter
import tiktoken


import re

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    
    token_count = len(enc.encode(text))
    
    # Heavy penalty for Table Placeholders to force split
    # If chunk_size is 4000, and we want max 1 table, 
    # then a table cost should be > 2000. Let's say 2500.
    # Pattern: [TABLE_X_Y]
    table_matches = re.findall(r"\[TABLE_[a-zA-Z0-9_]+_\d+\]", text)
    penalty = len(table_matches) * 500 
    
    return token_count + penalty

def process_splitting_all(data_dir: Path):
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    # Find all document folders (subdirectories)
    # Filter only directories that contain the text file
    dirs_to_process = [
        d for d in data_dir.iterdir() 
        if d.is_dir() and (d / "document_text_with_placeholders.txt").exists()
    ]
    
    if not dirs_to_process:
        print("No processed document folders found.")
        return

    print(f"Found {len(dirs_to_process)} document folders to process.")
    
    # Initialize Splitter
    # Base splitter: standard recursive splitter
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Table Aware Wrapper
    splitter = TableAwareSplitter(base_splitter=base_splitter)

    for doc_dir in dirs_to_process:
        print(f"\nProcessing: {doc_dir.name}")
        txt_path = doc_dir / "document_text_with_placeholders.txt"
        
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            # Split and Expand
            final_chunks = splitter.split_text(full_text)
            
            # Save Results
            output_path = doc_dir / "final_chunks.json"
            
            # Save as JSON list
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
                
            print(f"  - Generated {len(final_chunks)} chunks.")
            print(f"  - Saved to: {output_path}")
            
        except Exception as e:
            print(f"  - Error: {e}")

if __name__ == "__main__":
    target_dir = Path("./data/processed_experiment")
    process_splitting_all(target_dir)
