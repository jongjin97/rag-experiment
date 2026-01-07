from src.graph_rag.retriever import GraphRetriever

from src.graph_rag.retriever import GraphRetriever

def test_graph_rag(provider="openai"):
    print("Initializing Graph Retriever...")
    retriever = GraphRetriever(provider=provider)
    
    # 1. Local Search Test (Specific Entity)
    q1 = "HBM 메모리의 주요 특징과 시장 전망은?"
    print(f"\n\n[TEST 1] Local Search Query: {q1}")
    ans1 = retriever.query(q1, mode="local")
    print(f"Answer: {ans1}")
    
    # 2. Global Search Test (Broad Strategy)
    q2 = "삼성전자의 2024년 주주가치 제고 전략은 무엇인가?"
    print(f"\n\n[TEST 2] Global Search Query: {q2}")
    ans2 = retriever.query(q2, mode="global")
    print(f"Answer: {ans2}")

    # 3. Hybrid Search
    q3 = "삼성전자 DX부문의 주요 전략과 갤럭시 AI의 역할은?"
    print(f"\n\n[TEST 3] Hybrid Search Query: {q3}")
    ans3 = retriever.query(q3, mode="hybrid")
    print(f"Answer: {ans3}")

if __name__ == "__main__":
    import sys
    provider = "openai"
    if len(sys.argv) > 1:
        provider = sys.argv[1]
    
    print(f"Testing with Provider: {provider}")
    test_graph_rag(provider)
