# Samsung Report RAG Experiment Suite

이 프로젝트는 삼성전자 사업보고서 및 관련 기술 문서를 대상으로 **Retrieval-Augmented Generation (RAG)** 시스템의 성능을 극대화하기 위한 다양한 접근 방식을 연구하고 구현한 실험실입니다.

단순한 텍스트 검색부터 복잡한 지식 그래프(Knowledge Graph) 기반 추론까지, 서로 다른 아키텍처를 구현하고 비교 분석했습니다.

---

## 📚 Module Overview (비교 요약)

| Module | Name | Key Concept & Tech | Best Use Case | Complexity |
| :--- | :--- | :--- | :--- | :---: |
| **1. Optimization** | `rag_best_practices` | **Naive RAG Optimization** <br> (Hybrid Search + Reranking + Adaptive Chunking) | 일반적인 사실 검색, <br> 빠른 응답이 필요한 서비스 | ⭐ |
| **2. Optimization v2** | `rag_best_practices_v2` | **Table-Aware Advanced RAG** <br> (Table Restoration + HyDE + Cross-Encoder) | **복잡한 표가 포함된 문서**, <br> 오답률 0%에 근접해야 하는 정밀 서비스 | ⭐⭐ |
| **3. Light Graph** | `light_rag` | **Dual-Level Retrieval** <br> (Vector + 1-Hop Graph) | 핵심 개체(Entity)와 관계성 파악, <br> 비용 효율적인 그래프 검색 | ⭐⭐⭐ |
| **4. Light Graph (v2)** | `light_rag_v2` | **Graph Exp Integration** <br> (Sanitized Graph Data) | **높은 관련성(Relevancy)**이 필요한 <br> 질의응답, 데이터 무결성 강화 | ⭐⭐⭐ |
| **5. Deep Graph** | `graph_rag` | **Community Detection** <br> (Leiden Alg + Global Summaries) | 거시적인 트렌드 분석, <br> **높은 정확도의 사실(Fact) 검색**, <br> 복합적인 추론이 필요한 심층 질문 | ⭐⭐⭐⭐⭐ |
| **6. Deep Graph (Kor)** | `graph_rag_v2` | **Native Korean Graph** <br> (Full-scale extraction w/o translation) | 삼성전자 사업보고서 등 <br> 대규모 한국어 문서 정밀 분석 실험 | ⭐⭐⭐⭐⭐ |
| **7. Graph Experiment** | `graph_experiment` | **Table RAG & Topology Opt** <br> (PDF Tables, Batch API, Super-Node Logic) | 복잡한 표가 많은 문서, <br> 비용 효율적인 대규모 그래프 구축 | ⭐⭐⭐⭐⭐ |

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

## 2️⃣ RAG Best Practices v2 (`src/rag_best_practices_v2`)

재무제표 등 복잡한 **표(Table)** 데이터가 다수 포함된 문서를 대상으로, 정보 손실을 원천적으로 차단하고 검색 정확도를 극대화한 Advanced 파이프라인입니다.

- **핵심 혁신 (Key Innovations)**:
    - **Table Restoration Chunking**: 잘려나가는 표를 방지하기 위해, 파싱 단계에서 표를 분리한 후 청킹 시점에 완전한 **Markdown 형태로 복원**하여 삽입. (최적 청크: `512 Tokens`)
    - **Query Expansion (HyDE)**: LLM 가상 답변을 통한 쿼리 의미 확장으로, Hybrid 검색망의 넓은 재현율(Recall)을 유지하면서도 무관한 노이즈를 억제.
    - **Precision 대폭발 (Cross-Encoder)**: HyDE 기반 Hybrid로 긁어모은 문서들을 `bge-reranker-m3`로 1:1 교차 검증하여 완벽에 가까운 순위 재정렬 수행.
- **최종 검증 성능 (Ragas)**:
    - **[검색 품질 - 기존 데이터셋]**
        - **Context Relevance**: **0.969**
        - **Precision**: **0.872** (초기 모델 대비 **+13.6%p 폭등**)
        - **Recall**: **0.896**
    - **[생성 품질 - 신규 평가 데이터셋]**
        - **Faithfulness**: **0.7957** (검색 기반 사실성 확보)
        - **Answer Relevancy**: **0.7294**
        - 💡 **Insight**: 0점 케이스의 80.6%가 환각이 아닌 "문맥에서 정보를 찾을 수 없습니다"라는 정직한 응답으로, **엄격한 Hallucination 통제력**이 증명되었습니다.

👉 [자세히 보기](src/rag_best_practices_v2/README.md)

---

## 3️⃣ LightRAG (`src/light_rag`)

GraphRAG의 복잡도와 구축 비용을 줄이면서도 그래프의 이점을 취하기 위해 설계된 경량화 시스템입니다.

- **핵심 아키텍처 (Dual-Level)**:
    - **Low-Level**: 특정 Entity와 직접 연결된 관계(1-Hop)를 탐색하여 사실 관계 검증.
    - **High-Level**: 관계(Relation) 자체를 벡터화하여 추상적인 연결 고리 검색.
- **주요 특징**:
    - **Standardized GEXF**: 네트워크 표준 포맷인 GEXF를 통해 그래프 데이터 관리.
    - **Hybrid Retrieval**: Entity + Relation + Chunk 벡터를 모두 활용하여 답변 생성.
    - **Performance (100 Samples)**: 
        - **Faithfulness**: 0.76 (높은 신뢰성)
        - **Answer Relevancy**: 0.60
        - 기본 추출 로직 사용으로 인해 관련성에서 다소 한계 확인.

👉 [자세히 보기](src/light_rag/README.md)

---

## 4️⃣ LightRAG v2 (`src/light_rag_v2`)

LightRAG v1의 구조에 **Graph Experiment**의 고품질 데이터를 통합하여 **정답 관련성(Answer Relevancy)**을 대폭 개선한 고도화 모듈입니다.

- **개선 사항 (Improvements)**:
    - **Data Sanitization**: `Description` 및 `Type` 필드의 결측치를 자동 보정하여 데이터 파이프라인 안정화.
    - **Optimized Graph Data**: `Graph Experiment`에서 정제된(Topology Optimized) 그래프 데이터를 마이그레이션하여 활용.
- **Comparative Performance (v1 vs v2)**:
    - **Answer Relevancy**: **0.60 (v1) → 0.66 (v2)** (약 10% 향상)
    - 고품질의 그래프 컨텍스트가 주입됨에 따라 사용자의 질문 의도를 더 정확히 파악하고 답변함.
> ⚠️ **Critical Analysis**: `v2`의 점수 변동은 문서에 없는 답변을 거부하는 **엄격한 Hallucination 방지 메커니즘**과 **데이터셋의 외부 지식 혼입/평가 지표의 불안정성**에 기인합니다. (상세 분석은 [LightRAG v2 README](./src/light_rag_v2/README.md#%EF%B8%8F-dataset-constraints--insights-%ED%8F%89%EA%B0%80-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%EC%9D%98-%ED%95%9C%EA%B3%84) 참조)

👉 [자세히 보기](src/light_rag_v2/README.md)

---

## 5️⃣ GraphRAG (`src/graph_rag`)

Microsoft GraphRAG의 개념을 고도화하여 구현한 모듈로, 문서 집합 전체를 아우르는 **Global Context** 이해에 초점을 맞춥니다.

- **핵심 프로세스**:
    1.  **Graph Build**: LLM을 이용해 모든 엔티티와 관계를 추출.
    2.  **Community Detection**: Leiden 알고리즘으로 의미적 군집(Community) 형성.
    3.  **Summarization**: 각 커뮤니티를 계층적으로 요약하여 "거시적 지식" 생성.
- **기술적 해결**:
    - **Super Node 문제**: Top-K Pruning으로 토큰 폭발 방지.
    - **Cost Optimization**: OpenAI Batch API 도입으로 구축 비용 50% 절감.
- **성능**:
    - **Local Search**: **Answer Relevance 0.58**로 구체적인 사실 검색에서 가장 우수.
    - **Hybrid Search**: 대규모 평가에서 **Answer Relevance 0.57**을 기록하며 복합 추론에서도 일관된 고성능 입증.
    - (Baseline: Naive RAG 0.15)

👉 [자세히 보기](src/graph_rag/README.md)

---

## 6️⃣ GraphRAG v2 (Native Korean Experiment) (`src/graph_rag_v2`)

영어 프롬프트 기반의 v1과 달리, **순수 한국어 프롬프트**를 사용하여 4개의 삼성전자 사업보고서를 정밀 분석한 실험적 모듈입니다.

- **실험 배경 (Context)**:
    - 번역 과정을 배제하고 한국어 뉘앙스(예: '당사', '연결실체')를 직접 그래프에 반영하기 위해 설계되었습니다.
- **성능 (Ragas Automated)**:
    - **Faithfulness**: 0.75
    - **Answer Relevance**: 0.50 (v1 대비 다소 낮음)
- **주요 구조적 차이 (Key Structural Differences)**:
    - **Quantity Strategy**: 
        - **v1 (English)**: `Top 10 Limit` → **Unnormalized Graph** (중복 노드 다수 존재, 연결성 낮음)
        - **v2 (Korean)**: `No Limit` + `Strong Normalization` → **Dense Graph** (노드 수는 적지만 연결 밀도가 30% 더 높음, 정보 응집력 강화)
    - **Normalization**: v2는 '당사' 등의 대명사를 '삼성전자'로 치환하여 파편화된 정보를 하나로 통합함.

👉 [자세히 보기](src/graph_rag_v2/README.md)

---

## 7️⃣ Graph Experiment (`src/graph_experiment`)

**"GraphRAG v2"의 진화형**으로, 실제 서비스 레벨의 파이프라인 구축을 위해 **데이터 전처리(Table)**와 **그래프 품질(Topology)**, **비용 효율성(Batch API)**을 극한으로 최적화한 실험실입니다.

- **핵심 혁신 (Key Innovations)**:
    1.  **Table RAG**: 
        - PDF 내 복잡한 표를 **Multi-page Merging**, **Color-based Header Detection**으로 완벽 복원.
        - `[TABLE_ID]` 플레이스홀더와 **Table-Aware Chunking**으로 LLM에 테이블 문맥을 온전하게 전달.
    2.  **Topology Optimization**:
        - **Super Node Removal**: 삼성전자와 같은 초거대 노드를 탐지 시 제외하여, **Sub-Hub(사업부 등)** 중심의 커뮤니티 형성 유도.
        - **Prompt Engineering**: 고립 노드(Isolates) 71% 감소, 성게 현상(Sea Urchin) 방지.
    3.  **Local Search Retriever**:
        - 단순 키워드 매칭이 아닌 **LLM 기반 Entity Extraction**으로 시작 노드를 스마트하게 탐색.
    4.  **Cost Efficiency**:
        - **OpenAI Batch API**를 활용한 3단계 파이프라인(Prepare-Submit-Process)으로 구축 비용 50% 절감.

👉 [자세히 보기](src/graph_experiment/README.md)

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

### 2. RAG Best Practices v2 (Table & HyDE)
```bash
# 단계별 성능 축적 평가 (Colab 권장)
python -m src.rag_best_practices_v2.chunking
python -m src.rag_best_practices_v2.retrieval
python -m src.rag_best_practices_v2.hyde_experiment
python -m src.rag_best_practices_v2.reranking
```

### 3. LightRAG 실행
```bash
# 시스템 검증 (질의응답 테스트)
python src/light_rag/verify_system.py
```

### 4. LightRAG v2 실행 (New)
```bash
# 데이터 마이그레이션 (Graph Experiment -> LightRAG)
python -m src.light_rag_v2.indexer.process_batch

# Ragas 평가
python src/light_rag_v2/evaluate_ragas.py
```

### 5. GraphRAG 검색
```bash
# 로컬/글로벌/하이브리드 검색 테스트
python -m src.graph_rag.test_retrieval
```

### 6. Graph Experiment (New Strategy)
```bash
# 로컬 검색 (Graph Traversal)
python -m src.graph_experiment.retriever
```
