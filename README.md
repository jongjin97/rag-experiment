# Samsung Report RAG Experiment Suite

이 프로젝트는 삼성전자 사업보고서 및 관련 기술 문서를 대상으로 **Retrieval-Augmented Generation (RAG)** 시스템의 성능을 극대화하기 위한 다양한 접근 방식을 연구하고 구현한 실험실입니다.

단순한 텍스트 검색부터 복잡한 지식 그래프(Knowledge Graph) 기반 추론까지, 3가지 서로 다른 아키텍처를 구현하고 비교 분석했습니다.

---

## 📚 Module Overview (비교 요약)

| Module | Name | Key Concept & Tech | Best Use Case | Complexity |
| :--- | :--- | :--- | :--- | :---: |
| **1. Optimization** | `rag_best_practices` | **Naive RAG Optimization** <br> (Hybrid Search + Reranking + Adaptive Chunking) | 일반적인 사실 검색, <br> 빠른 응답이 필요한 서비스 | ⭐ |
| **2. Light Graph** | `light_rag` | **Dual-Level Retrieval** <br> (Vector + 1-Hop Graph) | 핵심 개체(Entity)와 관계성 파악, <br> 비용 효율적인 그래프 검색 | ⭐⭐⭐ |
| **3. Deep Graph** | `graph_rag` | **Community Detection** <br> (Leiden Alg + Global Summaries) | 거시적인 트렌드 분석, <br> 복합적인 추론이 필요한 심층 질문 | ⭐⭐⭐⭐⭐ |

---

## 1️⃣ RAG Best Practices (`src/rag_best_practices`)

기본적인 RAG 파이프라인의 각 단계(Chunking, Embedding, Retrieval, Reranking)를 실험적으로 최적화한 모듈입니다.

- **핵심 발견 (Key Findings)**:
    - **Chunking**: `256 Tokens`가 한국어 문서에서 정보 밀도와 검색 정확도 균형이 가장 좋음.
    - **Embedding**: `intfloat/multilingual-e5-large` 사용 시 Baseline 대비 4.5배 성능 향상.
    - **Retrieval**: BM25(키워드)와 Dense(의미)를 결합한 **Hybrid Search**가 필수적.
    - **Reranking**: **DLM(Cross-Encoder)** 도입 시 MRR 점수가 0.30 → 0.39로 대폭 개선됨.
- **권장 아키텍처**: Chunk 256 + Multilingual E5 + Hybrid Search + DLM Reranking (Top-5)

👉 [자세히 보기](src/rag_best_practices/README.md)

---

## 2️⃣ LightRAG (`src/light_rag`)

GraphRAG의 복잡도와 구축 비용을 줄이면서도 그래프의 이점을 취하기 위해 설계된 경량화 시스템입니다.

- **핵심 아키텍처 (Dual-Level)**:
    - **Low-Level**: 특정 Entity와 직접 연결된 관계(1-Hop)를 탐색하여 사실 관계 검증.
    - **High-Level**: 관계(Relation) 자체를 벡터화하여 추상적인 연결 고리 검색.
- **주요 특징**:
    - **Standardized GEXF**: 네트워크 표준 포맷인 GEXF를 통해 그래프 데이터 관리.
    - **Hybrid Retrieval**: Entity + Relation + Chunk 벡터를 모두 활용하여 답변 생성.
    - **Performance**: Ragas 평가 결과 **Faithfulness 0.66**, **Answer Relevancy 0.74** 달성.

👉 [자세히 보기](src/light_rag/README.md)

---

## 3️⃣ GraphRAG (`src/graph_rag`)

Microsoft GraphRAG의 개념을 고도화하여 구현한 모듈로, 문서 집합 전체를 아우르는 **Global Context** 이해에 초점을 맞춥니다.

- **핵심 프로세스**:
    1.  **Graph Build**: LLM을 이용해 모든 엔티티와 관계를 추출.
    2.  **Community Detection**: Leiden 알고리즘으로 의미적 군집(Community) 형성.
    3.  **Summarization**: 각 커뮤니티를 계층적으로 요약하여 "거시적 지식" 생성.
- **기술적 해결**:
    - **Super Node 문제**: Top-K Pruning으로 토큰 폭발 방지.
    - **Cost Optimization**: OpenAI Batch API 도입으로 구축 비용 50% 절감.
- **성능**: 복합 질문에 대해 **Answer Relevance 0.57**로 가장 우수한 추론 능력 입증.

👉 [자세히 보기](src/graph_rag/README.md)

---

## 🚀 Quick Start

각 모듈은 독립적으로 실행 및 테스트가 가능합니다.

### 1. Best Practices 실험
```bash
# 청크 사이즈 실험
python -m src.rag_best_practices.experiment_chunking
# 검색 전략 비교
python -m src.rag_best_practices.experiment_retrieval
```

### 2. LightRAG 실행
```bash
# 시스템 검증 (질의응답 테스트)
python src/light_rag/verify_system.py
```

### 3. GraphRAG 검색
```bash
# 로컬/글로벌/하이브리드 검색 테스트
python -m src.graph_rag.test_retrieval
```
