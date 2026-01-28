from src.graph_experiment.document_loader import load_documents_merged
from src.config import DATA_DIR
from pathlib import Path

DOC_PATH = DATA_DIR / "samsung" / "[삼성전자] 반기보고서(일반법인) (2025.08.14).pdf"
OUTPUT_FILE = Path("loaded_document_output.md")

if __name__ == "__main__":
    print(f"Loading document from {DOC_PATH} (Pages 150-160)...")
    documents = load_documents_merged(str(DOC_PATH), page_range=(150, 160))
    
    if documents:
        content = documents[0].page_content
        print(f"Loaded {len(content)} characters.")
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"Full content saved to {OUTPUT_FILE.absolute()}")
        print("\n--- Start of Content Preview ---")
        print(content[:500])
        print("--- End of Content Preview ---")
    else:
        print("No documents loaded.")