import pdfplumber
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

def table_to_markdown(data: List[List[str]]) -> str:
    """
    Converts a list of lists (table data) into a Markdown table string.
    Handles None values and multiline cell content.
    """
    if not data:
        return ""

    # Clean data: replace None with empty string, replace newlines with spaces
    cleaned_data = []
    for row in data:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                # Replace newlines within a cell to keep the markdown table structure valid
                cleaned_row.append(str(cell).replace("\n", " "))
        cleaned_data.append(cleaned_row)

    if not cleaned_data:
        return ""

    # Generate Markdown Table
    # Header
    header = cleaned_data[0]
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"

    # Rows
    for row in cleaned_data[1:]:
        markdown += "| " + " | ".join(row) + " |\n"

    return markdown

def extract_page_content_with_tables(page) -> str:
    """
    Extracts text and tables from a page, preserving their vertical order.
    1. Finds tables and their bounding boxes.
    2. Defines text regions (above, between, below tables).
    3. Extracts text from those regions.
    4. Converts tables to Markdown.
    5. Combines everything in order.
    """
    # 1. Find all tables
    tables = page.find_tables()
    
    # 2. If no tables, just return text
    if not tables:
        return page.extract_text() or ""

    # 3. Sort tables by vertical position (top) to process in order
    sorted_tables = sorted(tables, key=lambda t: t.bbox[1])
    
    page_width = page.width
    page_height = page.height
    
    content_parts = []
    current_y = 0

    for table in sorted_tables:
        # Table bounding box: (x0, top, x1, bottom)
        t_bbox = table.bbox
        t_top = t_bbox[1]
        t_bottom = t_bbox[3]

        # Region before the table (Text Region)
        if t_top > current_y:
            # Define bbox for text area: (0, current_y, page_width, t_top)
            # Use a small margin to avoid overlapping exact lines
            text_bbox = (0, current_y, page_width, t_top)
            try:
                # crop() might fail if the area is too small
                cropped_page = page.crop(text_bbox)
                text_segment = cropped_page.extract_text()
                if text_segment:
                    content_parts.append(text_segment)
            except Exception:
                # Ignore very small/invalid crops
                pass

        # The Table itself (Table Region)
        try:
            table_data = table.extract()
            md_table = table_to_markdown(table_data)
            content_parts.append(md_table)
        except Exception as e:
            print(f"Warning: Failed to extract table: {e}")

        # Update current Y position
        current_y = t_bottom

    # Region after the last table (Text Region)
    if current_y < page_height:
        text_bbox = (0, current_y, page_width, page_height)
        try:
            cropped_page = page.crop(text_bbox)
            text_segment = cropped_page.extract_text()
            if text_segment:
                content_parts.append(text_segment)
        except Exception:
            pass

    return "\n\n".join(content_parts)

def load_documents(path_input: Path) -> List[Document]:
    """
    Load PDF documents from a directory or a single file using pdfplumber.
    Tables are converted to Markdown and interleaved with text.
    """
    documents = []
    path_obj = Path(path_input)
    
    if not path_obj.exists():
        print(f"Path not found: {path_obj}")
        return []

    # Determine files to process
    if path_obj.is_file():
        files_to_process = [path_obj] if path_obj.suffix.lower() == ".pdf" else []
    elif path_obj.is_dir():
        files_to_process = list(path_obj.glob("*.pdf"))
    else:
        return []

    for file_path in files_to_process:
        print(f"Processing: {file_path.name}...")
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    content = extract_page_content_with_tables(page)
                    if content.strip():
                        # Create LangChain Document
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "page": i + 1,
                                "filename": file_path.name
                            }
                        )
                        documents.append(doc)
            print(f"  - Loaded {len(pdf.pages)} pages.")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            
    return documents

if __name__ == "__main__":
    # Test script included in main execution
    import sys
    
    # Simple test with one file if available
    test_dir = Path("./data/samsung")
    if test_dir.exists():
        docs = load_documents(test_dir)
        print(f"\nTotal loaded documents: {len(docs)}")
        
        # Print a sample with a table if possible
        print("\n--- Sample Content (First 500 chars of a page) ---")
        if docs:
            print(docs[0].page_content[:500])