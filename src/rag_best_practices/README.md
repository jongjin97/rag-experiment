# Samsung Business Report RAG Optimization

이 문서는 삼성전자 사업보고서 RAG 시스템의 성능을 극대화하기 위해 수행한 **Chunking, Retrieval, Reranking** 단계별 실험 결과와 최적화 전략을 정리한 기술 보고서입니다.

## 🏆 Executive Summary (최종 권장 아키텍처)

모든 실험 결과를 종합한 최적의 RAG 파이프라인 구성은 다음과 같습니다.

| Component | Recommendation | Reason |
|-----------|----------------|--------|
| **Embedding** | `intfloat/multilingual-e5-large` | 한국어 텍스트 문맥 이해도 우수 (Baseline 대비 4.5배 성능 향상) |
| **Chunking** | **256 Tokens** | 정보 밀도가 높고 검색 정확도가 가장 우수한 구간 |
| **Retrieval** | **Hybrid (Alpha=0.5)** | 용어 매칭(BM25)과 의미 검색(Dense)의 조화로 Hit Rate 1.4배 향상 |
| **Reranking** | **DLM (Cross-Encoder)** | 정확도가 최우선인 경우 필수 (MRR 0.30 -> 0.39). 단, 실시간성은 고려 필요. |

### 📈 단계별 성능 향상 요약 (Evolution of Performance)

전체 최적화 과정을 거치며 성능이 다음과 같이 단계적으로 향상되었습니다.

| 단계 (Phase) | 핵심 변경 사항 | Hit Rate | CTX Relevance | MRR | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0: Baseline** | English Model (Dense) | ~0.10 | 0.156 | Low | 한국어 이해 실패 |
| **P1: Embedding** | **Multilingual Model** | 0.327 | 0.704 | 0.249 | **성능 4.5배 급상승** (Foundation) |
| **P2: Retrieval** | **Hybrid Search** | 0.471 | 0.832 | 0.304 | Hit Rate 1.4배 향상 (Precision) |
| **P3: Reranking** | **DLM (BGE)** | **0.500** | **0.873** | **0.387** | **Final Polish** (Top-1 정확도 개선) |

---

## 1. 실험 환경 및 데이터셋
- **문서**: `[삼성전자] 반기보고서(일반법인) (2025.08.14).pdf` (금융/기술 복합 도메인)
- **평가 데이터**: GPT-4o로 생성한 고품질 QA Pair (104건)
- **평가 도구**: Ragas (Context Relevance), Custom Metrics (Hit Rate, MRR, Latency)

---

## 2. 실험 1: Chunking Strategy (청크 크기 최적화)

### 실험 내용
임베딩 모델별로 다양한 청크 사이즈(128~1024)가 검색 품질(Context Relevance)에 미치는 영향을 측정했습니다.

### 주요 결과
| Model | Chunk Size | Mean Score | 결론 |
|-------|------------|------------|------|
| **English Model** | 256 | 0.156 | 한국어 이해도 부족으로 성능 매우 저조 |
| **Multilingual Model** | 128 | 0.651 | 문맥이 너무 짧아 정보 손실 발생 |
| **Multilingual Model** | **256** | **0.704** | **최적 성능**. 정보 밀도와 검색 정확도의 균형 |
| **Multilingual Model** | 512 | 0.702 | 256과 유사하나 노이즈 포함 빈도 증가 |
| **Multilingual Model** | 1024 | 0.683 | 불필요한 정보가 섞이며 관련성 하락 |

> **Insight**: 한국어 RAG에서는 **Multilingual 모델 필수**이며, **256~512 토큰**이 가장 적합함.

---

## 3. 실험 2: Retrieval Strategy (검색 기법 비교)

### 실험 내용
최적 청크(256) 환경에서 **Dense Only(Baseline)**, **Hybrid(BM25+Dense)**, **HyDE** 방식의 성능을 비교했습니다.

### 주요 결과 (Top-K=5)
| Method | CR (Relevance) | Hit Rate | MRR | Latency |
|--------|----------------|----------|-----|---------|
| **Baseline (Dense)** | 0.714 | 0.327 | 0.249 | **0.19s** |
| **Hybrid (α=0.5)** | **0.832** | **0.471** | **0.304** | 0.21s |
| **HyDE** | 0.702 | 0.308 | 0.245 | 3.50s |

> **Insight**: 
> 1. **Hybrid Search**가 Dense 단독 사용 대비 **Hit Rate를 약 1.4배(0.32->0.47) 개선**함. 금융 보고서의 고유명사/수치 검색에 BM25가 효과적임.
> 2. **HyDE**는 생성 비용(3.5초) 대비 성능 이득이 없어 탈락.

---

## 4. 실험 3: Reranking Optimization (재정렬 정확도)

### 실험 내용
Hybrid 검색으로 상위 **20개**를 추출한 뒤, 정밀한 **Reranker**로 상위 **5개**를 재정렬했을 때의 효과를 검증했습니다. (DLM vs TILDE)

### 주요 결과 (Top 20 -> 5)
| Method | Type | CR (Relevance) | Hit Rate | MRR | Latency (CPU) |
|--------|------|----------------|----------|-----|---------------|
| **Baseline** | Hybrid Top-5 | 0.822 | 0.471 | 0.304 | **1.42s** |
| **DLM (BGE)** | Cross-Encoder | **0.873** | **0.500** | **0.387** | 25.52s |
| **TILDE** | Query Likelihood | 0.538 | 0.115 | 0.058 | 7.68s |

> **Insight**:
> 1. **DLM (BGE-M3)**는 **MRR을 0.30 -> 0.39로 크게 향상**시킴. 가장 정확한 문서를 최상단에 배치하는 능력이 탁월함.
> 2. 단, **CPU 환경에서는 속도가 매우 느림(25초)**. 실서비스 적용 시 GPU가 필수적이거나, 오프라인 배치 분석용으로 적합함.
> 3. **TILDE**는 한국어 미지원으로 인해 성능이 크게 하락하여 사용 불가.

---

## 5. 결론 및 제언

1. **기본 배포 (Real-time Service)**
   - **구성**: Chunk 256 + Multilingual Emb + **Hybrid Search (Top-5)**
   - **이유**: 평균 0.2~1초 내의 빠른 응답 속도와 준수한 정확도(Hit Rate 0.47) 제공.

2. **고성능 모드 (High Precision / Offline Analysis)**
   - **구성**: Chunk 256 + Multilingual Emb + Hybrid Search (Top-20) + **DLM Reranking (Top-5)**
   - **이유**: 응답 시간은 길어지지만, **가장 높은 정확도(Hit Rate 0.50, MRR 0.39)**를 보장하여 환각(Hallucination) 최소화. 

## 6. 사용 방법 (Reproduction)

```bash
# 1. 청크 실험
python -m src.rag_best_practices.experiment_chunking

# 2. 검색 실험 (Baseline vs Hybrid vs HyDE)
python -m src.rag_best_practices.experiment_retrieval

# 3. 리랭킹 실험 (Baseline vs DLM vs TILDE)
python -m src.rag_best_practices.experiment_reranking
```
