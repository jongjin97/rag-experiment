# Advanced RAG Lab: 삼성전자 사업보고서 기반의 차세대 검색 시스템 구축

> **"단순 검색을 넘어, 지식 그래프를 통한 거시적 통찰력까지"**
>
> RAG(Retrieval-Augmented Generation)의 성능 한계를 극복하기 위해, **Chunking 최적화**부터 **Dual-Level LightRAG**, **Deep GraphRAG**, 그리고 **Graph Topology Optimization**까지 4단계로 진화하는 검색 시스템을 구축하고 비교 분석한 프로젝트입니다.

---

## 📸 프로젝트 미리보기 (Screenshots)

### 1. LightRAG Dual-Level Retrieval
![LightRAG Architecture](https://github.com/user-attachments/assets/lightrag-diagram.png) *Dual-Level Retrieval: Low-Level(Entity)과 High-Level(Relation)을 결합하여 정밀한 답변을 생성합니다.*

### 2. GraphRAG Global Context
![GraphRAG Pipeline](https://github.com/user-attachments/assets/graphrag-pipeline.png) *Community Summarization: 문서 전체의 맥락을 군집화하여 "삼성전자 2024 전략"과 같은 거시적 질문에 답합니다.*

---

## 📖 프로젝트 소개 (Overview)

금융 및 기술 문서(예: 삼성전자 사업보고서)는 전문 용어와 복잡한 인과 관계가 얽혀 있어, 기존의 단순 RAG(Vector Search) 방식으로는 정확한 답변을 얻기 어렵습니다.

이 프로젝트는 **"어떻게 하면 RAG가 더 똑똑해질 수 있을까?"** 라는 질문에서 시작하여, 다음 4가지 핵심 과제를 해결하고자 했습니다.

1.  **한국어 텍스트 및 복잡한 표(Table) 구조 파괴**: 플레이스홀더를 활용한 테이블 마크다운 복원 및 청킹 크기 최적화 (`rag_best_practices_v2`)
2.  **검색 재현율(Recall) 및 정밀도(Precision) 한계**: HyDE + 하이브리드 검색 + Cross-Encoder 리랭킹 통합 파이프라인 구축 (`rag_best_practices_v2`)
3.  **단편적 정보 검색의 한계**: 지식 그래프를 도입하여 개체 간의 관계를 파악 (`light_rag`)
4.  **거시적 질문(Global QA) 실패**: 문서 전체를 아우르는 군집 요약 및 토폴로지 최적화 (`graph_rag`, `graph_experiment`)

단순한 구현을 넘어, 각 단계별로 성능 지표(Faithfulness, MRR 등)를 정량적으로 측정하고 최적의 아키텍처를 도출했습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 | 사용 목적 |
| :--- | :--- | :--- |
| **언어** | Python 3.10+ | 메인 개발 언어 |
| **LLM & Embedding** | OpenAI GPT-4o, Multilingual-E5 | 추론, 데이터 생성, 벡터 임베딩 |
| **Vector DB** | ChromaDB | 텍스트 청크 및 임베딩 벡터 저장 |
| **Graph** | NetworkX, Leiden Algorithm, **Neo4j** | 지식 그래프 구축, 저장 및 커뮤니티 탐지 |
| **Framework** | LangChain, Ragas | RAG 파이프라인 구축 및 성능 평가 |
| **Data Processing** | **OpenAI Batch API**, **pdfplumber** | 대규모 그래프 추출 비용 절감 (50%↓), 정밀한 표(Table) 전처리 |

---

## 🏗️ 핵심 기능 및 아키텍처 (Key Features)

프로젝트는 문제 해결의 깊이에 따라 4가지 모듈로 구성됩니다.

### 1. Advanced RAG Pipeline (문서 전처리 및 검색 최적화 v2)
> *"테이블은 살리고, 검색망은 넓게, 정답은 예리하게"*

- **Table Restoration**: 복잡한 표를 `[TABLE_0_1]` 형태의 플레이스홀더로 우선 분리해 텍스트 구조 파편화를 방지하고, 청킹(Chunking) 후 마크다운으로 원상 복구시켜 완벽한 정보 보존 
- **Adaptive Chunking**: 테이블 마크다운 환경에서 `512 Token` 구간이 파편화를 막으면서도 정보 밀도를 유지하는 최적 크기임을 증명
- **HyDE + Hybrid Search**: Dense(의미)와 BM25(키워드) 결합(5:5)에 **가상 답변 생성(HyDE)**을 심어, 숨겨진 문맥까지 포획하며 **재현율(Recall) 0.877** 극대화
- **Cross-Encoder Reranking**: 넓게 가져온 문서들의 노이즈를 1:1로 정밀히 채점해 차단, **정밀도(Precision)를 폭발적으로 상향(0.872)**시키고 **Context Relevance 0.969** 달성

### 2. LightRAG (경량화 그래프)
> *"비용 효율적인 지식 그래프 검색"*

- **Dual-Level Retrieval**:
    - **Low-Level**: "HBM3E의 스펙은?" → Entity 검색으로 정확한 사실 검증.
    - **High-Level**: "메모리 사업부와 모바일 사업부의 관계는?" → Relation 검색으로 맥락 파악.
- **Efficient Indexing**: GEXF 표준 포맷을 활용하여 그래프와 벡터 DB를 효율적으로 동기화.

### 3. GraphRAG (심층 그래프)
> *"숲을 보는 거시적 통찰력"*

- **Community Detection**: Leiden 알고리즘으로 문서 내 의미적 군집(Community)을 발견.
- **Global Summarization**: 각 군집을 계층적으로 요약하여, 특정 키워드 없이도 전체 맥락을 묻는 질문("올해의 리스크 요인은?")에 답변 가능.
- **Batch Processing**: OpenAI Batch API를 도입하여 구축 비용을 **50% 절감**하고 토큰 제한 문제를 해결.

### 4. Graph Experiment (토폴로지 최적화 & Table RAG)
> *"데이터 품질이 곧 검색 품질이다"*

- **Table RAG**: PDF 내 복잡한 표를 **Multi-page Merging**, **Color-based Header Detection**으로 완벽 복원하고 `[TABLE_ID]` 플레이스홀더로 문맥 손실 방지.
- **Topology Optimization**:
    - **Quantitative Diagnosis**: 고립 노드(Isolates)를 71% 감소시키고 거대 컴포넌트 비율을 97.5%로 향상.
- **Local Search Retriever**: LLM 기반 Entity Extraction으로 시작 노드를 스마트하게 탐색하여 정확도 향상.

---

## 💻 핵심 코드 (Core Code Snippets)

### 1. Dual-Level Retrieval Logic (`light_rag/retriever.py`)
사용자의 질문 의도에 따라 **Entity(구체적)**와 **Relation(추상적)** 정보를 동시에 검색하여 결합하는 핵심 로직입니다.

```python
async def retrieve_hybrid(self, query: str, top_k: int = 5) -> str:
    """
    Entity(Low)와 Relation(High) 정보를 결합하여 풍부한 맥락을 제공합니다.
    """
    # 1. 텍스트 청크 검색 (Original Text)
    chunk_context = await self.retrieve_chunks(query, top_k=top_k)
    
    # 2. 엔티티 검색 (Specific Facts)
    low_level_context = await self.retrieve_low_level(query, top_k=top_k)
    
    # 3. 관계 검색 (Broader Relationships)
    high_level_context = await self.retrieve_high_level(query, top_k=top_k)
    
    # 문맥 결합
    return (
        f"=== Original Text Context ===\n{chunk_context}\n\n"
        f"=== Entity Context ===\n{low_level_context}\n\n"
        f"=== Relationship Context ===\n{high_level_context}"
    )
```

### 2. Community Detection & Summarization (`graph_rag/detect_communities.py`)
NetworkX 그래프에서 Leiden 알고리즘을 수행하여 의미 있는 군집(Community)을 찾아내는 과정입니다.

```python
def detect_communities(graph: nx.Graph) -> Dict[int, List[str]]:
    """
    Leiden 알고리즘을 사용하여 그래프 내의 커뮤니티를 감지합니다.
    """
    # NetworkX 그래프를 igraph로 변환 (Leiden 알고리즘 지원)
    ig_graph = ig.Graph.from_networkx(graph)
    
    # 커뮤니티 탐지 수행
    partition = leidenalg.find_partition(
        ig_graph, 
        leidenalg.ModularityVertexPartition
    )
    
    # 결과 매핑
    communities = {}
    for i, nodes in enumerate(partition):
        # 소규모 군집 필터링 (노이즈 제거)
        if len(nodes) > MIN_COMMUNITY_SIZE:
             communities[i] = [ig_graph.vs[node]['_nx_name'] for node in nodes]
             
    return communities
```

### 3. Hybrid Search Implementation (`rag_best_practices/retrieval.py`)
BM25(키워드)와 Dense(의미) 검색을 가중치(Alpha) 기반으로 결합하여 검색 성능을 극대화한 로직입니다.

```python
def get_ensemble_retriever(chunk_size: int = 256, alpha: float = 0.5):
    """
    Returns a Hybrid Retriever (Vector + BM25).
    Alpha: Weight for Sparse (BM25). 
    """
    # 1. BM25 Retriever (Sparse)
    bm25 = get_bm25_retriever(chunk_size)
    
    # 2. Dense Retriever (Vector)
    vectorstore = get_vectorstore(chunk_size)
    dense = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. Ensemble (Hybrid)
    print(f"Initializing Hybrid Retriever with Alpha={alpha}")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[alpha, 1.0 - alpha]
    )
    return ensemble_retriever
```

### 4. Cost-Effective Batch Indexing (`light_rag/indexer/prepare_batch.py`)
수천 번의 LLM 호출을 OpenAI Batch API 포맷으로 변환하고, 토큰 제한(Limit)에 맞춰 파일을 자동 분할하는 전처리 파이프라인입니다.

```python
def prepare_lightrag_batch():
    """
    문서를 청킹하고 OpenAI Batch API 요청 포맷(JSONL)으로 변환합니다.
    토큰 제한(1.5M)을 초과하지 않도록 자동으로 배치를 분할합니다.
    """
    chunks = split_documents(docs, chunk_size=1000)
    
    for i, chunk in enumerate(chunks):
        # 토큰 수 계산 (Tiktoken)
        tokens_est = count_tokens(chunk.page_content) + 1000
        
        # Request Body 생성
        request_body = {
            "custom_id": f"chunk_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": chunk.page_content}],
                "tools": [graph_extraction_tool]
            }
        }
        
        # 배치 파일 용량(Token Limit) 체크 및 분할 저장
        if current_tokens + tokens_est > TOKENS_PER_FILE_LIMIT:
             save_batch_file(current_batch_requests, file_index)
             file_index += 1
             current_batch_requests = []
             current_tokens = 0
             
        current_batch_requests.append(request_body)
        current_tokens += tokens_est
```

---

## 🔧 트러블슈팅 및 성능 개선 (Troubleshooting)

### 1. Super Node로 인한 토큰 폭발 (Token Overflow)
- **문제**: 'Samsung Electronics' 같은 중심 노드는 수천 개의 엣지를 가져, 검색 시 Context Window를 초과하는 문제 발생.
- **해결**: **Top-K Neighbor Pruning** 도입. 연결 강도(Weight)가 높은 상위 20개 이웃만 선별적으로 로드하도록 로직을 수정하여, 토큰 사용량을 90% 절감하면서도 핵심 정보는 유지했습니다.

### 2. 그래프 구축 비용 문제 (Cost Efficiency)
- **문제**: 수천 개의 텍스트 청크를 실시간 API로 처리하여 그래프를 구축할 때 비용과 시간이 과도하게 소요됨.
- **해결**: **OpenAI Batch API** 파이프라인(`prepare_batch.py` -> `submit` -> `process`)을 구축. 비동기 일괄 처리를 통해 비용을 50% 절감하고 Rate Limit 문제를 해결했습니다.

### 3. 한국어 문맥 이해 부족 (Language Mismatch)
- **문제**: 초기 Baseline 모델(English Dense) 사용 시 한국어 금융 문서의 검색 정확도가 **Hit Rate 0.10** 수준으로 매우 저조했습니다.
- **해결**: **Multilingual-E5-Large** 모델 교체 및 **256 Token Chunking** 전략 적용. 이를 통해 Hit Rate를 0.32(3배↑)로 끌어올렸으며, 1024 토큰 등 긴 청크보다 짧고 밀도 높은 청크가 한국어 RAG에 더 유리함을 증명했습니다.

### 4. 정확도와 응답 속도의 딜레마 (Precision vs Latency)
- **문제**: 많은 문서를 검색(Top-K=100)하면 정답 포함 확률은 높지만 환각(Hallucination)이 늘고 LLM 비용이 증가하며, 적게 검색하면 정답을 놓치는 문제가 발생했습니다.
- **해결**: **Reranking (2-Stage Retrieval)** 도입. Hybrid Search로 20개를 빠르게 추린 후, **DLM Cross-Encoder**로 상위 5개를 정밀 재정렬하는 파이프라인을 구축하여 **MRR을 0.30에서 0.39로 개선**하고 Top-1 정확도를 극대화했습니다. 

### 5. 그래프 품질 문제: 성게 현상 (Sea Urchin Effect)
- **문제**: '2024년', '매출' 같은 날짜/수치가 노드로 추출되면서, 수천 개의 엣지가 연결된 의미 없는 **허브(Hub)**가 생성되고 탐색 효율이 저하됨.
- **해결**: **Prompt Engineering**을 통해 날짜와 수치를 노드가 아닌 **엣지의 속성(Property)**으로 변환하도록 강제. 이 결과 **노이즈 노드를 47% 감소**시키고 그래프의 정보 밀도를 높였습니다.
    - *Before*: (삼성전자)-[HAS]-(2024년), (SK하이닉스)-[HAS]-(2024년)
    - *After*: (삼성전자)-[COMPETES_WITH {date: 2024년}]-(SK하이닉스)

---

## 📊 실험 결과 (Performance Benchmark)

자체 구축한 QA 데이터셋과 **Ragas** 프레임워크를 활용하여 두 가지 측면에서 정량적 성능 평가를 수행했습니다.

### 1. RAG 파이프라인 최적화 (Retrieval Performance)
문서 검색 단계의 품질을 Context 지표로 추적 및 고도화한 결과입니다. (`rag_best_practices_v2` 기준)

| 아키텍처 | CR (Relevance) | Precision | Recall | 특징 및 효과 |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Dense)** | 0.929 | 0.808 | 0.779 | 의미적 문맥 탐색에는 강하나 고유명사/수치에 상대적으로 약함 |
| **Table RAG + Hybrid** | 0.931 | 0.723 | 0.858 | 표 보존 및 하이브리드 결합으로 재현율(Recall) 대폭 상승 |
| **HyDE + Hybrid** | 0.940 | 0.736 | 0.877 | 문맥 확장(HyDE)을 통해 숨겨진 맥락 획득, 전 지표 동반 상승 |
| **전체 복합 + Reranking** | **0.969** | **0.872** | **0.896** | **파이프라인 최강 조합. 노이즈 완벽 차단으로 정밀도 대폭발** |

### 2. GraphRAG 구조 비교 (Generation Performance)
각종 그래프 RAG 기법이 최종 생성해 내는 답변의 신뢰도(Faithfulness)와 관련도를 측정한 결과입니다.

| 아키텍처 | Faithfulness | Answer Relevancy | 특징 |
| :--- | :---: | :---: | :--- |
| **Naive RAG** | 0.98 | 0.37 | 팩트 검색엔 강하나, 맥락 파악 능력 부족 |
| **LightRAG** | 0.66 | **0.75** | **높은 답변 관련성**. 하이브리드 검색의 효과 입증 |
| **GraphRAG Global** | **1.00** | 0.48 | 전체 요약에 기반하여 허구(Hallucination)가 없음 |
| **Graph Local** | **0.99** | 0.50 | Naive RAG보다 빠르고 정확함 (Entity 중심 검색) |
| **Graph Hybrid** | 0.95 | **0.57** | 복합적인 질문에 대해 가장 높은 답변 품질을 보임 |

> **인사이트**: 
> 1. 검색 엔진(Retriever) 차원에서는 플레이스홀더로 **표(Table)를 보존**한 뒤, **HyDE + Hybrid + Reranking**을 결합하는 것이 타협 없는 최고의 검색 정밀도(`Context Precision 0.872`)를 보장합니다.
> 2. 맥락 파악 및 구조적 릴레이션(Relation) 추론이 필요할 때는 **LightRAG**나 **Graph Hybrid**가 답변 수준(Relevance)을 월등히 높입니다.
> 3. 연간 트렌드 요약 등 거시적 통찰력에는 100%의 사실성을 자랑하는 **GraphRAG Global** 모드가 가장 안전한 선택지입니다.

---

## 🚀 회고 (Retrospective)

- **배운 점**: RAG의 성능은 단순히 좋은 모델을 쓰는 것이 아니라, **'데이터를 어떻게 구조화하느냐'**에 달려있음을 깨달았습니다. 특히 지식 그래프를 통해 비정형 텍스트에 구조를 입힘으로써, LLM이 더 논리적인 추론을 할 수 있게 돕는 과정이 인상 깊었습니다.
- **아쉬운 점**: 그래프 구축 시간이 오래 걸려 실시간 데이터 업데이트(Incremental Update)에 대한 실험이 부족했습니다. 향후에는 **Neo4j**를 도입하여 새로운 문서가 추가될 때 전체 그래프를 다시 만들지 않고 부분적으로 업데이트하는 **Dynamic Graph Update** 기능을 추가하고 싶습니다.
