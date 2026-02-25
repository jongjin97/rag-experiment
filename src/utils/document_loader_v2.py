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

def clean_table(table_data: List[List[str]]) -> List[List[str]]:
    """
    Removes columns that are completely empty/None across all rows.
    """
    if not table_data:
        return []
    
    num_cols = len(table_data[0])
    cols_to_keep = []
    
    for c in range(num_cols):
        # Check if column c is empty in all rows
        is_empty = True
        for row in table_data:
            val = row[c]
            if val is not None and str(val).strip():
                is_empty = False
                break
        if not is_empty:
            cols_to_keep.append(c)
            
    # Reconstruct table
    new_table = []
    for row in table_data:
        new_row = [row[c] for c in cols_to_keep]
        new_table.append(new_row)
        
    return new_table

def extract_page_elements(page) -> List[dict]:
    """
    Extracts text and tables from a page as structured elements.
    Returns a list of dictionaries: {'type': 'text'|'table', 'content': str|List[List[str]], 'row_colors': [...], 'bbox': tuple}
    """
    # 1. Find all tables
    tables = page.find_tables()
    
    # 2. If no tables, just return text
    if not tables:
        text = page.extract_text()
        if text and text.strip():
            return [{
                'type': 'text',
                'content': text.strip(),
                'bbox': (0, 0, page.width, page.height)
            }]
        return []

    # 3. Filter overlapping tables (sub-tables)
    # If a table is fully contained within another, skip it.
    filtered_tables = []
    # Sort by size (area) descending first to prioritize larger tables when checking containment?
    # No, usually we want the biggest table.
    # Let's sort by Y first.
    sorted_tables_by_y = sorted(tables, key=lambda t: t.bbox[1])
    
    # Simple N^2 check is fine for small N tables per page
    for i, t1 in enumerate(sorted_tables_by_y):
        is_contained = False
        t1_area = (t1.bbox[2]-t1.bbox[0]) * (t1.bbox[3]-t1.bbox[1])
        
        for j, t2 in enumerate(sorted_tables_by_y):
            if i == j:
                continue
            
            # Check if t1 is inside t2
            # Use small tolerance
            if (t1.bbox[0] >= t2.bbox[0]-1 and t1.bbox[1] >= t2.bbox[1]-1 and
                t1.bbox[2] <= t2.bbox[2]+1 and t1.bbox[3] <= t2.bbox[3]+1):
                
                # Verify t2 is actually larger/different
                t2_area = (t2.bbox[2]-t2.bbox[0]) * (t2.bbox[3]-t2.bbox[1])
                if t2_area > t1_area:
                    is_contained = True
                    break
                    
        if not is_contained:
            filtered_tables.append(t1)
            
    sorted_tables = filtered_tables
    
    page_width = page.width
    page_height = page.height
    
    elements = []
    current_y = 0

    for table in sorted_tables:
        # Table bounding box: (x0, top, x1, bottom)
        t_bbox = table.bbox
        t_top = t_bbox[1]
        t_bottom = t_bbox[3]

        # Region before the table (Text Region)
        if t_top > current_y + 1: # Buffer
            # Define bbox for text area: (0, current_y, page_width, t_top)
            # Use a small margin to avoid overlapping exact lines
            text_bbox = (0, current_y, page_width, t_top)
            try:
                # crop() might fail if the area is too small
                cropped_page = page.crop(text_bbox)
                text_segment = cropped_page.extract_text()
                if text_segment and text_segment.strip():
                    elements.append({
                        'type': 'text',
                        'content': text_segment.strip(),
                        'bbox': text_bbox
                    })
            except Exception:
                # Ignore very small/invalid crops
                pass

        # The Table itself (Table Region)
        try:
            table_data = table.extract()
            # Clean the table immediately to avoid issues with spacer columns
            cleaned_data = clean_table(table_data)

            # Extract row colors
            row_colors = []
            for row in table.rows:
                color = get_row_bg_color(row.bbox, page.rects, page.height)
                row_colors.append(color)
            
            if cleaned_data:
                elements.append({
                    'type': 'table',
                    'content': cleaned_data,
                    'row_colors': row_colors,
                    'bbox': t_bbox
                })
        except Exception as e:
            print(f"Warning: Failed to extract table: {e}")

        # Update current Y position
        current_y = max(current_y, t_bottom)

    # Region after the last table (Text Region)
    if current_y < page_height:
        text_bbox = (0, current_y, page_width, page_height)
        try:
            cropped_page = page.crop(text_bbox)
            text_segment = cropped_page.extract_text()
            if text_segment and text_segment.strip():
                elements.append({
                    'type': 'text',
                    'content': text_segment.strip(),
                    'bbox': text_bbox
                })
        except Exception:
            pass

    return elements

def get_header_rows(table_data, row_colors):
    """
    Helper to extract header rows based on color.
    Returns (header_rows_data, header_row_count)
    """
    if not row_colors or not table_data:
        return [table_data[0]], 1
    
    first_color = row_colors[0]
    if first_color is None:
        return [table_data[0]], 1
        
    count = 0
    for color in row_colors:
        if color == first_color:
            count += 1
        else:
            break
            
    if count > 0:
        return table_data[:count], count
    return [table_data[0]], 1

def merge_document_elements(all_elements: List[dict]) -> str:
    """
    Merges consecutive tables and converts everything to a single markdown string.
    """
    if not all_elements:
        return ""
        
    merged_elements = []
    
    # Initial pass to merge consecutive tables
    if not all_elements:
        return ""
        
    current_element = all_elements[0]
    
    for next_element in all_elements[1:]:
        # Check if both are tables and can be merged
        if (current_element['type'] == 'table' and 
            next_element['type'] == 'table'):
            
            table1 = current_element['content']
            colors1 = current_element.get('row_colors')
            
            table2 = next_element['content']
            colors2 = next_element.get('row_colors')
            
            # Check 1: Body Continuation
            # If table2 starts with None color, it's a body.
            # FORCE MERGE regardless of columns (User request)
            is_body_continuation = False
            if colors2 and len(colors2) > 0 and colors2[0] is None:
                is_body_continuation = True

            # Check 2: Header Continuation (Split Header)
            # If table1 ended with Color AND table2 starts with Color -> Header Split
            is_header_continuation = False
            if (colors1 and len(colors1) > 0 and colors1[-1] is not None and
                colors2 and len(colors2) > 0 and colors2[0] is not None):
                is_header_continuation = True
            
            cols_match = (len(table1[0]) == len(table2[0]))

            # Priority 1: Body Continuation (Force Merge)
            if is_body_continuation:
                 current_element['content'].extend(table2)
                 if colors1 is not None and colors2 is not None:
                     current_element['row_colors'].extend(colors2)
                 elif colors1 is not None:
                     current_element['row_colors'].extend([None] * len(table2))
                 continue

            # Priority 2: Header Continuation & Repeated Header (Require Column Match)
            if cols_match:
                if is_header_continuation:
                    # Append directly
                    current_element['content'].extend(table2)
                    if colors1 is not None and colors2 is not None:
                        current_element['row_colors'].extend(colors2)
                    elif colors1 is not None:
                        current_element['row_colors'].extend([None] * len(table2))
                    continue

                # Check 3: Repeated Header (Removed by user request)
                # We do not merge based on repeated headers anymore.
                pass
                
        # If not merged, push current and update
        merged_elements.append(current_element)
        current_element = next_element
        
    merged_elements.append(current_element)
    
    # Convert to string
    final_output = []
    for elem in merged_elements:
        if elem['type'] == 'text':
            final_output.append(elem['content'])
        elif elem['type'] == 'table':
            final_output.append(table_to_markdown(elem['content'], elem.get('row_colors')))
            
    return "\n\n".join(final_output).strip()

def load_documents_merged(path_input: Path, page_range: Optional[tuple] = None) -> List[Document]:
    """
    Load PDF documents, merging ALL pages of each file into a single Document object.
    This preserves cross-page context for better chunking.
    
    :param path_input: Path to PDF file or directory.
    :param page_range: Optional tuple (start_page, end_page) to process only specific pages. 0-indexed.
                      If provided, e.g. (0, 5), pages 0 to 4 will be processed.
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
            all_page_elements = []
            with pdfplumber.open(file_path) as pdf:
                # Determine pages to process
                pages_to_process = pdf.pages
                if page_range:
                    start, end = page_range
                    # Ensure range is valid
                    start = max(0, start)
                    end = min(len(pdf.pages), end)
                    pages_to_process = pdf.pages[start:end]
                    print(f"  - Processing pages {start} to {end-1} (of {len(pdf.pages)})")

                for i, page in enumerate(pages_to_process):
                    # if i != 151: # This line was for debugging, removing for general use
                    #     continue
                    # print("testing") # This line was for debugging, removing for general use
                    page_elements = extract_page_elements(page)
                    all_page_elements.extend(page_elements)
            
            combined_content = merge_document_elements(all_page_elements)
            
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
        docs = load_documents_merged(test_dir, page_range=(0, 10))
        print(f"\nTotal loaded documents: {len(docs)}")
        
        # Print a sample with a table if possible
        print("\n--- Sample Content (First 500 chars of a page) ---")
        if docs:
            print(docs[0].page_content[:10000])