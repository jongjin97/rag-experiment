import pdfplumber
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

def vertical_overlap(a0, a1, b0, b1):
    overlap = max(0, min(a1, b1) - max(a0, b0))
    return overlap / max(1, min(a1 - a0, b1 - b0))

def row_bbox_to_pdf_coords(row_bbox, page_height):
    x0, top, x1, bottom = row_bbox
    pdf_y0 = page_height - bottom
    pdf_y1 = page_height - top
    return x0, pdf_y0, x1, pdf_y1

def get_row_bg_color(row_bbox, rects, page_height):
    rx0, ry0, rx1, ry1 = row_bbox_to_pdf_coords(row_bbox, page_height)

    candidates = []
    for r in rects:
        if not r.get("fill"):
            continue

        # X축은 넉넉하게 겹치기만 하면 OK
        x_overlap = not (r["x1"] < rx0 or r["x0"] > rx1)

        # Y축은 "의미 있는 overlap"만 인정
        y_overlap_ratio = vertical_overlap(
            ry0, ry1,
            r["y0"], r["y1"]
        )

        if x_overlap and y_overlap_ratio > 0.5:
            candidates.append((r, y_overlap_ratio))

    if not candidates:
        return None

    # overlap 가장 큰 rect 선택
    best_rect = max(candidates, key=lambda x: x[1])[0]
    return best_rect.get("non_stroking_color")



def table_to_markdown(data: List[List[str]], row_colors: Optional[List[Optional[tuple]]] = None) -> str:
    """
    Converts a list of lists (table data) into a Markdown table string.
    Handles None values and multiline cell content.
    If row_colors is provided, uses it to identify header rows (consecutive rows with same non-None color at the top).
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

    # Determine header rows
    header_rows = []
    body_rows = []
    
    has_colored_header = False
    if row_colors and len(row_colors) == len(cleaned_data):
        first_color = row_colors[0]
        if first_color is not None:
            # Find consecutive rows with the same color
            header_end_idx = 0
            for color in row_colors:
                if color == first_color:
                    header_end_idx += 1
                else:
                    break
            
            # If we found at least 1, treating them as header
            # (If all rows have same color, it might be a styling choice, but we still treat as header + body? 
            #  Actually if ALL are header, where is body? 
            #  Let's limit: valid header shouldn't be the WHOLE table usually, but MD table needs at least 1 header row)
            if header_end_idx > 0:
                 header_rows = cleaned_data[:header_end_idx]
                 body_rows = cleaned_data[header_end_idx:]
                 has_colored_header = True
    
    # Fallback if no color formatting found
    if not has_colored_header:
        header_rows = [cleaned_data[0]]
        body_rows = cleaned_data[1:]

    # Generate Markdown Table
    markdown = ""
    
    # Header Rows
    for row in header_rows:
        markdown += "| " + " | ".join(row) + " |\n"
        
    # Separator (Use column count from the first header row)
    if header_rows:
        num_cols = len(header_rows[0])
        markdown += "| " + " | ".join(["---"] * num_cols) + " |\n"

    # Body Rows
    for row in body_rows:
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

            row_colors = []
            for row in table.rows:
                color = get_row_bg_color(row.bbox, page.rects, page.height)
                row_colors.append(color)
            print("row colors: ", row_colors)
            md_table = table_to_markdown(table_data, row_colors)
            print("md table: ", md_table)
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

def load_documents_merged(path_input: Path) -> List[Document]:
    """
    Load PDF documents, merging ALL pages of each file into a single Document object.
    This preserves cross-page context for better chunking.
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
        print(f"Processing (Merged): {file_path.name}...")
        try:
            full_text = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i != 151:
                        continue
                    print("testing")
                    content = extract_page_content_with_tables(page)
                    if content.strip():
                        full_text.append(content)
            
            combined_content = "\n\n".join(full_text)
            
            if combined_content.strip():
                # Create Single LangChain Document for the entire file
                doc = Document(
                    page_content=combined_content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "total_pages": len(pdf.pages)
                    }
                )
                documents.append(doc)
                print(f"  - Loaded {len(pdf.pages)} pages into 1 unified document.")
                
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            
    return documents

if __name__ == "__main__":
    # Test script included in main execution
    import sys
    
    # Simple test with one file if available
    test_dir = Path("./data/samsung/[삼성전자] 반기보고서(일반법인) (2025.08.14).pdf")
    if test_dir.exists():
        docs = load_documents_merged(test_dir)
        print(f"\nTotal loaded documents: {len(docs)}")
        
        # Print a sample with a table if possible
        print("\n--- Sample Content (First 500 chars of a page) ---")
        if docs:
            print(docs[0].page_content[:10000])