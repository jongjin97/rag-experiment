
import json
from pathlib import Path
from typing import List, Tuple, Dict
import pdfplumber
import tiktoken
from src.graph_experiment.document_loader import extract_page_elements, get_header_rows, table_to_markdown

def merge_elements_with_placeholders(all_elements: List[dict]) -> Tuple[str, Dict[str, List[List[str]]]]:
    """
    Merges tables considering the same split logic as document_loader.
    Instead of outputting Markdown inline, it replaces the table with [TABLE_X] 
    and returns a dictionary of tables.
    """
    if not all_elements:
        return "", {}
        
    merged_elements = []
    
    # Initial pass to merge consecutive tables (Logic verification needed if copying from document_loader)
    # We ideally want to reuse the specific logic in document_loader's merge_document_elements,
    # but that function returns string.
    # So we duplicate the merge loop here to ensure identical behavior but different output structure.
    
    current_element = all_elements[0]
    
    # --- MERGE LOGIC START (Copied from document_loader.py to maintain consistency) ---
    for next_element in all_elements[1:]:
        if (current_element['type'] == 'table' and 
            next_element['type'] == 'table'):
            
            table1 = current_element['content']
            colors1 = current_element.get('row_colors')
            
            table2 = next_element['content']
            colors2 = next_element.get('row_colors')
            
            # Check 1: Body Continuation (Force Merge for None start)
            is_body_continuation = False
            if colors2 and len(colors2) > 0 and colors2[0] is None:
                is_body_continuation = True

            # Check 2: Header Continuation (Split Header)
            is_header_continuation = False
            if (colors1 and len(colors1) > 0 and colors1[-1] is not None and
                colors2 and len(colors2) > 0 and colors2[0] is not None):
                is_header_continuation = True
            
            cols_match = (len(table1[0]) == len(table2[0]))

            # Priority 1: Body Continuation
            if is_body_continuation:
                 current_element['content'].extend(table2)
                 if colors1 is not None and colors2 is not None:
                     current_element['row_colors'].extend(colors2)
                 elif colors1 is not None:
                     current_element['row_colors'].extend([None] * len(table2))
                 continue

            # Priority 2: Header Continuation
            if cols_match:
                if is_header_continuation:
                    current_element['content'].extend(table2)
                    if colors1 is not None and colors2 is not None:
                        current_element['row_colors'].extend(colors2)
                    elif colors1 is not None:
                        current_element['row_colors'].extend([None] * len(table2))
                    continue

                # Priority 3: Repeated Header (User Removed)
                pass
                
        # If not merged
        merged_elements.append(current_element)
        current_element = next_element
        
    merged_elements.append(current_element)
    # --- MERGE LOGIC END ---

    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        print("Warning: gpt-4o encoding not found, falling back to cl100k_base")
        enc = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS = 1024 # Adjustable threshold

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    final_text_parts = []
    tables_storage = {}
    table_counter = 0

    for elem in merged_elements:
        if elem['type'] == 'text':
            final_text_parts.append(elem['content'])
        elif elem['type'] == 'table':
            original_table_id = f"TABLE_{table_counter}"
            
            # Split into Header and Body
            header_rows_data, h_count = get_header_rows(elem['content'], elem.get('row_colors'))
            header = elem['content'][:h_count]
            body = elem['content'][h_count:]

            # Chunking Logic
            chunks = []
            current_chunk_body = []
            
            # Recalculate header tokens to be safe
            header_str = table_to_markdown(header)
            header_tokens = count_tokens(header_str)
            current_chunk_tokens = header_tokens 

            for row in body:
                # Approximate row cost
                row_str = "| " + " | ".join([str(c).replace("\n", " ") if c else "" for c in row]) + " |\n"
                row_tokens = count_tokens(row_str)
                
                if current_chunk_tokens + row_tokens > MAX_TOKENS and current_chunk_body:
                    # Finalize current chunk
                    chunks.append(current_chunk_body)
                    # Start new chunk
                    current_chunk_body = [row]
                    current_chunk_tokens = header_tokens + row_tokens
                else:
                    current_chunk_body.append(row)
                    current_chunk_tokens += row_tokens
            
            # Add last chunk
            if current_chunk_body:
                chunks.append(current_chunk_body)
            
            # If no body (only header?), keep as one chunk
            if not body and not chunks:
                chunks = [[]]

            # Save Chunks
            chunk_placeholders = []
            for i, chunk_body in enumerate(chunks):
                chunk_id = f"{original_table_id}_{i}"
                
                # Reconstruct full table for this chunk (Header + Chunk Body)
                full_chunk_content = header + chunk_body # List + List
                
                tables_storage[chunk_id] = {
                    "id": chunk_id,
                    "header": header,
                    "body": chunk_body,
                    "markdown": table_to_markdown(full_chunk_content, None) # No color needed for just MD gen
                }
                chunk_placeholders.append(f"[{chunk_id}]")
            
            # Insert Placeholders
            # User request: Only provide the last chunk index. 
            # The system should interpret [TABLE_X_N] as "Load TABLE_X_0 through TABLE_X_N".
            if chunk_placeholders:
                final_text_parts.append(chunk_placeholders[-1])
            
            table_counter += 1

    return "\n\n".join(final_text_parts), tables_storage

def process_batch(input_path: Path, output_dir: Path, page_range=None):
    if not input_path.exists():
        print(f"Path not found: {input_path}")
        return

    # Determine files to process
    files_to_process = []
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process = list(input_path.glob("*.pdf"))
    else:
        print(f"Invalid input path: {input_path}")
        return

    if not files_to_process:
        print("No PDF files found to process.")
        return

    print(f"Found {len(files_to_process)} PDF(s) to process.")

    for pdf_path in files_to_process:
        print(f"\nProcessing file: {pdf_path.name}")
        
        # Create output subdirectory for this file
        file_output_dir = output_dir / pdf_path.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            all_page_elements = []
            with pdfplumber.open(pdf_path) as pdf:
                pages = pdf.pages
                if page_range:
                     start, end = page_range
                     start = max(0, start)
                     end = min(len(pages), end)
                     pages = pages[start:end]
                
                print(f"  - Extracting pages...")
                for page in pages:
                    # Reusing extraction logic
                    extracted = extract_page_elements(page)
                    all_page_elements.extend(extracted)
            
            # Process content
            print(f"  - Merging and chunking tables...")
            text_with_placeholders, tables_map = merge_elements_with_placeholders(all_page_elements)
            
            # Save Text
            text_out_path = file_output_dir / "document_text_with_placeholders.txt"
            with open(text_out_path, "w", encoding="utf-8") as f:
                f.write(text_with_placeholders)
            print(f"  - Saved text to: {text_out_path}")
            
            # Save Tables
            tables_out_path = file_output_dir / "extracted_tables.json"
            with open(tables_out_path, "w", encoding="utf-8") as f:
                json.dump(tables_map, f, ensure_ascii=False, indent=2)
            print(f"  - Saved {len(tables_map)} tables to: {tables_out_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

if __name__ == "__main__":
    # Batch processing directory
    input_location = Path("./data/samsung")
    output_location = Path("./data/processed_experiment")
    
    # Process all files (no page limit for real run, or keep it for testing?)
    # Removing page_range to process full documents as per user implication "files to process"
    process_batch(input_location, output_location, page_range=None)
