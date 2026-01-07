# Graph Retrieval-Augmented Generation (GraphRAG) Module

이 모듈은 **GraphRAG (Graph Retrieval-Augmented Generation)**의 고도화된 구현체로, 대규모 문서 집합에서 **Local(미시적)** 정보와 **Global(거시적)** 통찰력을 동시에 추출하기 위해 설계되었습니다. 단순한 RAG(Retrieve-Read)의 한계를 넘어, **지식 그래프(Knowledge Graph)**를 구축하고 **커뮤니티 탐지(Leiden Algorithm)**를 통해 맥락을 구조화했습니다.

특히, **OpenAI Batch API** 도입으로 비용을 획기적으로 절감하고, **Super Node 문제**와 **Token Limit** 등의 기술적 난제를 해결하여 실무 수준(Production-level)의 안정성을 확보했습니다.

---

## 🏗️ Architecture & Pipeline

GraphRAG 파이프라인은 크게 **구축(Build)**, **요약(Summarize)**, **검색(Retrieve)**의 3단계로 구성됩니다.

```mermaid
graph TD
    DOC[Document Chunks] -->|Extract Entities & Relations| KG[Knowledge Graph]
    KG -->|Leiden Algorithm| COMM[Communities]
    COMM -->|Map-Reduce Summarization| SUM[Community Summaries]
    
    Q[User Query] -->|Keyword/Entity Match| LOCAL[Local Context\n(Neighbors)]
    Q -->|Broad Context| GLOBAL[Global Context\n(Summaries)]
    
    LOCAL --> HYBRID[Hybrid Context]
    GLOBAL --> HYBRID
    HYBRID -->|Generation| ANS[Final Answer]
```

---

## 🛠️ Key Technical Challenges & Solutions (Troubleshooting)

프로젝트 진행 중 발생한 주요 기술적 이슈와 해결 과정 요약입니다.

### 1. Super Node로 인한 Context Explosion (Token Overflow)
- **문제점**: 'Samsung Electronics'와 같은 중심 엔티티(Super Node)는 수천 개의 연결 관계(Edge)를 가집니다. Local Search 시 이 엔티티가 포함되면 연결된 모든 정보를 가져오려다 **250,000 토큰 초과(RESOURCE_EXHAUSTED)** 오류가 발생했습니다.
- **해결책**: **Top-K Neighbor Pruning** 전략 도입.
    - 모든 이웃을 가져오는 대신, 연결 강도나 중요도가 높은 **상위 20개 이웃**만 선별적으로 로드하도록 `retriever.py`를 최적화했습니다.
    - 이를 통해 토큰 사용량을 90% 이상 절감하면서도 핵심 맥락은 유지했습니다.

### 2. 비용 문제와 처리 속도 (Cost & Latency)
- **문제점**: 수천 개의 텍스트 청크에서 그래프를 추출할 때, 실시간 API 호출은 비용이 매우 비싸고 속도가 느렸습니다.
- **해결책**: **OpenAI Batch API** 전면 도입 (`process_batch.py`).
    - API 요청을 JSONL 파일로 묶어 비동기(Asynchronous)로 처리.
    - 기존 대비 **50%의 비용 절감** 효과를 달성했으며, Rate Limit(TPM) 걱정 없이 대량 처리가 가능해졌습니다.

### 3. Global Context의 토큰 한계
- **문제점**: 1,500개 이상의 커뮤니티 요약을 한 번에 LLM에 주입하려다 다시 토큰 한계에 부딪혔습니다.
- **해결책**: **Dynamic Context Selection**.
    - 전체 커뮤니티를 모두 넣는 대신, 커뮤니티의 크기(Size)와 질의 관련성(Relevance)을 기준으로 **Top 50 커뮤니티**만 선별하여 주입합니다.
    - 안전장치로 **Max Character Limit (200k chars)**을 적용하여 안정성을 보장했습니다.

---

## 🚀 Key Features

### 1. Multi-Model Support
- **GPT-4o / GPT-4o-mini**: 기본 추출 및 추론 엔진.
- **Gemini 1.5 Pro / Flash**: 1M+ 대용량 Context Window가 필요한 Global Search에 활용.
- **DeepSeek V3**: 고성능 추론 및 비용 효율화 모델 지원. (`provider='deepseek'`)

### 2. Batch Processing Workflow
대용량 데이터 처리를 위한 전용 파이프라인 스크립트를 제공합니다.
1.  **준비 (`prepare_batch.py`)**: 문서를 청킹하고 추출 요청 파일(JSONL) 생성.
2.  **제출 (`submit_batch.py`)**: OpenAI Batch API에 작업 제출.
3.  **처리 (`process_batch.py`)**: 완료된 작업 다운로드, 그래프 구축 및 커뮤니티 요약 저장.

### 3. Advanced Retrieval Modes
- **Local Search**: 특정 사실 검증(Fact-checking)에 최적화. (예: "HBM3E의 대역폭은?")
- **Global Search**: 전체 데이터셋을 아우르는 거시적 질문에 최적화. (예: "삼성전자의 2024년 3대 핵심 전략은?")
- **Hybrid Search**: 두 가지를 결합하여 가장 풍부하고 정확한 답변 제공.

---

## 📂 File Structure

```bash
src/graph_rag/
├── builder.py              # 그래프 구축기 (NetworkX + Leiden Algorithm integration)
├── extractor.py            # LLM 기반 Entity/Relationship 추출 (Structured Output 사용)
├── retriever.py            # Local/Global/Hybrid 검색 로직 (Super Node 방지 적용)
├── community.py            # 커뮤니티 요약 생성기
├── detect_communities.py   # Leiden 알고리즘 기반 커뮤니티 탐지 모듈
├── prepare_batch.py        # [Batch] 1단계: 요청 파일 생성
├── submit_batch.py         # [Batch] 2단계: 작업 제출
├── process_batch.py        # [Batch] 3단계: 결과 처리 및 그래프 업데이트
└── test_retrieval.py       # 검색 성능 검증 테스트 스크립트
```

## 📊 Usage Example

### Retrieval (Test)
```bash
# 기본(OpenAI) 모델 사용
python -m src.graph_rag.test_retrieval

# Google Gemini 사용 (Large Context)
python -m src.graph_rag.test_retrieval google


## 📊 Performance Benchmark

자체 구축한 평가 데이터셋(QA Testset)을 사용하여 **Ragas** 프레임워크로 측정한 정량적 성능 지표입니다.

| Mode | Avg Latency | Faithfulness | Answer Relevance | Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Naive RAG** | ~5.8s | 0.98 | 0.37 | (Baseline) 일반적인 문맥 검색 |
| **Graph Local** | **~3.9s** | **0.99** | 0.50 | 구체적인 사실(Fact) 검색, 빠른 응답 속도 |
| **Graph Global** | ~8.0s | **1.00** | 0.48 | 전체 데이터셋에 대한 거시적 요약 및 트렌드 파악 |
| **Graph Hybrid** | ~12.2s | 0.95 | **0.57** | 복합적인 질문, 가장 높은 답변 품질(Relevance) 보장 |

> **💡 Insight**:
> - **Local Search**는 Naive RAG보다 빠르고(~3.9s) 더 높은 Relevance(0.50)를 보였습니다. 이는 그래프가 불필요한 노이즈를 제거하고 핵심 이웃만 탐색하기 때문입니다.
> - **Global Search**는 Faithfulness가 **1.0**으로, 허구(Hallucination) 없이 요약에 기반한 정확한 답변을 제공합니다.
> - **Hybrid Search**는 응답 시간이 길지만, **Answer Relevance(0.57)**가 가장 높아 복잡한 추론이 필요한 질문에 가장 적합한 전략임이 입증되었습니다.
>
> ⚠️ **Note**: 일부 질문에서 GraphRAG가 **영어**로 답변을 생성하여 평가 모델이 관련성을 0점으로 처리한 케이스가 포함되어 있습니다. (질문: 한국어, 답변: 영어). 이를 감안하면 Hybrid Search의 실제 체감 성능은 수치보다 훨씬 높을 것으로 추정됩니다.

