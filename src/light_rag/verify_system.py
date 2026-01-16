import asyncio
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.light_rag.lightrag import LightRAG

async def main():
    print("Initializing LightRAG...")
    rag = LightRAG()
    
    query = "HBM3E 12단의 대역폭은 얼마인가?"
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    # Test Hybrid Retrieval (Generation)
    # print("\nGenerating Answer (Hybrid Mode)...")
    # answer = await rag.query(query, mode="hybrid")
    # print("\n=== Answer ===")
    # print(answer)
    
    # Optional: Inspect Context
    context = await rag.retriever.retrieve_hybrid(query)
    print("\n=== Retrieved Context ===")
    print(context)

if __name__ == "__main__":
    asyncio.run(main())
