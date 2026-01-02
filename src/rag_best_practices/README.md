# RAG Best Practices: Chunking Experiment

이 문서는 삼성전자 사업보고서 RAG 시스템의 성능 최적화를 위한 **청크 사이즈(Chunk Size) 실험 결과**를 정리한 것입니다.

## 1. 실험 목적
- 한국어 금융/기술 문서(삼성전자 사업보고서)에 가장 적합한 청크 사이즈 도출
- 임베딩 모델 변경에 따른 성능 변화 검증 (English vs Multilingual)
- **평가 지표**: Ragas `ContextRelevance` (질문과 검색된 문맥 간의 연관성 점수)

## 2. 실험 환경
- **데이터셋**: `data/eval_dataset.json` (GPT-4o로 생성된 25개 QA Pair)
- **문서**: `[삼성전자] 반기보고서(일반법인) (2025.08.14).pdf`
- **검색기**: ChromaDB (Top-k=5)
- **평가 도구**: Ragas (LLM-based Evaluation)

## 3. 실험 결과

### V1: 영어 특화 모델 사용 (Original)
* **Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **설명**: 영문 데이터에 최적화된 모델로, 한국어 텍스트 임베딩 성능이 매우 저조함.

| Chunk Size | Mean Score | Std Dev |
|------------|------------|---------|
| 128        | 0.091      | 0.230   |
| 256        | 0.156      | 0.273   |
| 512        | 0.147      | 0.270   |
| 1024       | 0.168      | 0.303   |
> **분석**: 전체적으로 점수가 0.1대로 매우 낮음. 검색된 문맥이 질문과 무관한 경우가 대다수임.

### V2: 다국어 모델 변경 (Improved)
* **Model**: `BAAI/bge-m3` (또는 동급의 Multilingual 모델)
* **설명**: 한국어를 포함한 다국어 지원이 강력한 모델로 변경 후 재실험.

| Chunk Size | Mean Score | Std Dev |
|------------|------------|---------|
| 128        | 0.651      | 0.384   |
| **256**    | **0.704**  | 0.383   |
| **512**    | **0.702**  | 0.418   |
| 1024       | 0.683      | 0.413   |
> **분석**:
> 1.  모델 변경만으로 성능이 **약 4.5배 향상** (0.15 -> 0.70)
> 2.  **최적 사이즈**: **256 ~ 512** 토큰 구간에서 가장 높은 성능을 보임.
> 3.  **1024 이상**: 문맥에 불필요한 정보(Noise)가 섞이면서 점수가 소폭 하락하는 경향.

## 4. 결론 및 제언
1.  **모델 선정**: 한국어 RAG 시스템 구축 시 **Multilingual 임베딩 모델(예: BGE-M3, KoBERT 등) 사용이 필수적**임.
2.  **최적 청크 사이즈**: **256 ~ 512 토큰**이 정보 손실을 최소화하면서도 검색 정확도(Precision)를 유지하는 최적 구간임.
3.  **향후 계획**: 
    - 256 사이즈를 기본으로 채택하되, `ParentDocumentRetriever` (Small-to-Big) 전략을 적용하여 **검색은 256으로, 생성은 512~1024 문맥**을 제공하는 하이브리드 방식 고려.

## 5. 실행 방법
```bash
# 실험 실행 (결과 재현)
python -m src.rag_best_practices.chunking
```

## 6. 검색 전략 실험 (Retrieval Strategy)

청크 사이즈(256 tokens)를 고정한 상태에서 다양한 검색 기법을 비교 실험하였습니다.

### 실험 설정
* **Baseline**: Dense Retriever (ChromaDB, `intfloat/multilingual-e5-large`)
* **Hybrid**: BM25 (Sparse) + Dense (Ensemble Retriever)
* **HyDE**: Hypothetical Document Embeddings (LLM이 가상의 답변 생성 후 검색)
* **Metrics**:
    * **Context Relevance**: 질문과 검색 결과의 의미적 연관성 (Ragas)
    * **Hit Rate**: 정답 문맥(Ground Truth)이 상위 5개 내에 포함될 확률
    * **MRR (Mean Reciprocal Rank)**: 정답 문맥의 순위 역수 평균
    * **Latency**: 평균 검색 소요 시간 (초)

### 실험 결과 (Top-K=5)

| Method | Alpha (BM25 Weight) | Context Relevance | Hit Rate | MRR | Latency (s) |
|--------|---------------------|-------------------|----------|-----|-------------|
| **Baseline (Dense)** | - | 0.714 | 0.327 | 0.249 | **0.196** |
| **Hybrid** | 0.3 | 0.829 | 0.471 | 0.292 | 0.223 |
| **Hybrid** | **0.5** | **0.832** | **0.471** | **0.304** | 0.215 |
| **Hybrid** | 0.7 | 0.827 | 0.471 | 0.308 | 0.218 |
| **HyDE** | - | 0.702 | 0.308 | 0.245 | 3.498 |

### 분석 및 결론
1.  **Hybrid Search 우수성 입증**:
    *   Dense Only 대비 **Hit Rate가 약 1.4배 향상** (0.33 -> 0.47).
    *   금융 보고서 특성상 정확한 용어(Keyword) 매칭이 중요하여 BM25의 기여도가 큼.
    *   **Optimal Alpha**: 0.5 (오차 범위 내에서 0.3~0.7 모두 유사하게 우수함).
2.  **HyDE의 한계**:
    *   Baseline과 유사하거나 약간 낮은 성능을 보임.
    *   **Latency**: LLM 생성 과정으로 인해 3.5초가 소요되어 실시간 검색에는 부적합할 수 있음.
3.  **최종 전략 선정**:
    *   **Hybrid Search (Alpha=0.5)**를 최종 검색 전략으로 채택.
    *   검색 속도 저하(0.02초 차이)는 미미하면서 성능 향상은 확실함.
