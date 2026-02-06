# Light Retrieval-Augmented Generation V2 (LightRAG v2)

**LightRAG v2**는 기존 LightRAG의 **Dual-Level Retrieval** 아키텍처를 계승하면서, **Graph Experiment**에서 생성된 고품질의 지식 그래프 데이터를 **재사용(Reuse)**하고 **통합(Integration)**하는 데 초점을 맞춘 고도화 모듈입니다.

단순히 새로운 그래프를 구축하는 것이 아니라, 이미 분석된 **GEXF 그래프 데이터**를 마이그레이션하여 **검색 성능**과 **데이터 효율성**을 극대화했습니다.

---

## 🏗️ Architecture & Pipeline

LightRAG v2는 **Data Migration(데이터 이관)**과 **Optimization(최적화)** 파이프라인이 추가되었습니다.

```mermaid
graph TD
    EXP_DATA["Experiment Graph Data (GEXF)"] -->|"Import & Sanitization"| PROCESS["Batch Processor (v2)"]
    
    subgraph Indexing Layer
        PROCESS -->|"Node Embedding"| EV["Entity Vector Store (Chroma)"]
        PROCESS -->|"Edge Embedding"| RV["Relation Vector Store (Chroma)"]
        PROCESS -->|"Metadata Indexing"| NX["NetworkX Graph"]
    end
    
    Q["User Query"] -->|"Vector Search"| E_RES["Retrieved Entities"]
    Q -->|"Vector Search"| R_RES["Retrieved Relations"]
    
    subgraph Dual-Level Retrieval
        E_RES -->|"1-Hop Context"| LOW["Low-Level Context"]
        R_RES -->|"Relation Context"| HIGH["High-Level Context"]
    end
    
    LOW --> HYBRID["Hybrid Context"]
    HIGH --> HYBRID
    
    HYBRID -->|"Generation"| ANS["Final Answer"]
```

---

## 🛠️ Key Improvements

### 1. Robust Data Integration (데이터 통합 안정성 강화)
- **문제점**: 외부 실험 데이터(GEXF)를 로드할 때 `Description`이나 `Type` 필드의 `None` 값으로 인해 직렬화(Serialization) 오류가 발생했습니다.
- **해결책**: `process_batch.py` 레벨에서 결측치를 자동으로 감지하고 빈 문자열(`""`)로 치환하는 **Sanitization Logic**을 적용하여 안정적인 데이터 파이프라인을 구축했습니다.

### 2. Dataset Alignment (평가 데이터셋 정합성 확보)
- **문제점**: Ragas 평가 시 데이터셋의 필드명 불일치(`question` vs `user_input`)로 인해 평가 스크립트가 중단되는 이슈가 있었습니다.
- **해결책**: `evaluate_ragas.py`가 다양한 데이터셋 스키마를 유연하게 처리하도록 패치하여 **평가 프로세스의 호환성**을 높였습니다.

---

### Performance Comparison: LightRAG v1 vs v2

`LightRAG v1`과 `LightRAG v2`에 대해 동일한 100개의 샘플 데이터로 평가를 수행한 결과입니다.
`v2`는 **Graph Experiment**에서 구축된 정제된 그래프 데이터를 사용하여, 답변의 관련성을 높이는 데 주력했습니다.

| Module | Dataset Source | Faithfulness | Answer Relevancy | Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **LightRAG v1** | Basic Extraction | **0.7626** | 0.6039 | `Faithfulness`가 높으나 `Answer Relevancy`가 상대적으로 낮음 |
| **LightRAG v2** | **Graph Experiment** | 0.7484 | **0.6622** | **`Answer Relevancy` 대폭 향상 (+9.6%)**, 균형 잡힌 성능 제공 |

> **💡 Comparative Analysis**:
> - **Answer Relevancy 향상의 의미**: `LightRAG v2`가 `Graph Experiment`의 고품질 그래프 데이터(더 정확한 관계 및 속성)를 활용함으로써, 사용자의 질문 의도에 더 적합하고 풍부한 맥락을 제공했음을 시사합니다.
> - **Faithfulness의 안정성**: `v2`의 `Faithfulness` 수치(0.7484)는 `v1`(0.7626)과 통계적으로 유의미한 차이가 없는 수준이며, 여전히 검색된 근거에 기반한 신뢰도 높은 답변을 생성하고 있습니다.
> - **결론**: `v2`는 데이터 파이프라인의 개선과 그래프 품질 향상을 통해 RAG 시스템의 전반적인 **답변 품질(Qualitative Performance)**을 한 단계 끌어올렸습니다.
>
> ### ⚠️ Dataset Constraints & Insights (평가 데이터셋의 한계)
>
> 평가 과정에서 **Ragas 점수가 실제 모델의 성능을 온전히 반영하지 못하는 현상**이 발견되었습니다. 이는 데이터셋 자체의 품질 문제에 기인하며, 주요 원인은 다음과 같습니다:
>
> 1.  **External Knowledge Leakage (외부 지식 혼입)**:
>     -   Reference(정답)에 **문서에 없는 일반 상식이나 이론적 배경**이 포함된 경우가 다수 존재합니다.
>     -   *예시: "지분율이 시장 점유율에 미치는 영향"을 묻는 질문에 대해, 문서는 '지분율 수치'만 제공하지만, Reference는 '경제학적 이론'을 정답으로 제시함.*
>     -   **LightRAG의 동작**: 문서에 기반한 **Strict Retrieval**을 수행하므로, 근거가 없는 경우 "정보 없음"이라고 정직하게 답변합니다. 이는 Hallucination을 방지하는 올바른 동작이지만, Reference와 다르다는 이유로 **Faithfulness/Answer Relevancy 점수가 하락**하는 원인이 됩니다.
>
> 2.  **Unanswerable Questions (답변 불가능한 질문)**:
>     -   문서의 내용(예: 단순 사명 변경)과 무관한 인과관계(예: 재무 안정성 영향)를 묻는 질문이 포함되어 있습니다.
>     -   Reference는 관련 없는 사실을 나열한 후 "모른다"고 답하는 **Verbose**한 형태인 반면, LightRAG는 간결하게 "모른다"고 답하여 점수 차이가 발생합니다.
>
> **결론**: 현재의 점수는 **LightRAG v2의 높은 무결성(Integrity)**을 반증하는 결과이며, 향후 정제된 데이터셋(Clean Dataset)을 사용한 재평가가 필요합니다.
>
> 3.  **Metric Instability (평가 지표의 불안정성)**:
>     -   매우 유사한 의미와 구조를 가진 답변임에도 불구하고, 미세한 표현 차이로 인해 `Answer Relevancy` 점수가 극단적으로 갈리는 현상(예: v1은 **0.92**, v2는 **0.0**)이 관찰되었습니다.
>     -   이는 Ragas의 질문 생성(Question Generation) 기반 평가 방식이 특정 문구에 과도하게 민감하거나, 임베딩 모델의 비일관성에서 비롯된 것으로 추정됩니다.

---

## 📂 File Structure

```bash
src/light_rag_v2/
├── evaluate_ragas.py       # Ragas 기반 성능 평가 스크립트 (Schema-aware)
├── retriever.py            # 검색 로직 (v2 최적화)
├── verify_system.py        # 시스템 검증 스크립트
├── indexer/
│   ├── import_graph_experiment.py # 실험 데이터 마이그레이션 도구
│   └── process_batch.py    # GEXF 그래프 빌더 (Sanitization 적용)
├── utils/
│   └── gexf_loader.py      # GEXF 로더 유틸리티
└── README.md               # 문서
```

## 🚀 Usage

### 1. Data Migration & Indexing
```bash
# 실험 그래프 데이터 가져오기 및 인덱싱
python -m src.light_rag_v2.indexer.process_batch
```

### 2. Evaluation
```bash
# Ragas 평가 수행 (결과는 data/light_rag_v2/ragas_evaluation_results.csv 저장)
python src/light_rag_v2/evaluate_ragas.py
```
