# Graph RAG Performance Improvement Experiments

현재 지식 그래프(Knowledge Graph)를 활용한 RAG 시스템의 성능을 극대화하기 위한 실험을 진행 중입니다.
본 실험의 첫 번째 단계는 **문서 전처리(Pre-processing)** 과정, 특히 **테이블(Table) 데이터 처리**의 정확성을 검증하는 것입니다. 그래프 생성 전 데이터의 무결성을 확보하는 것이 목표입니다.

## Experiment 1: Table Extraction Strategy Verification

문서 파싱 및 구조화된 데이터(테이블) 추출을 위해 두 가지 주요 라이브러리의 성능을 비교 분석하였습니다.

### 1. `pdfplumber` (Current Selection)
- **Status**: 현재 사용 중
- **결과**:
  - **높은 감지율**: 문서 내의 테이블을 누락 없이 거의 대부분 찾아내는 우수한 성능을 보임.
- **이슈 (Known Issues)**:
  - **비정형 구조 오인식**: 테이블 내 특정 셀의 텍스트가 길어져 줄바꿈이 일어나는경우 이를 같은 컬럼의 내용이 아닌 **새로운 컬럼(New Column)**으로 오판하여 테이블 구조를 왜곡하는 현상이 확인됨.

### 2. `docling`
- **Status**: 실험적 도입 후 보류
- **결과 (주요 문제점)**:
  - **느린 처리 속도**: OCR(광학 문자 인식) 기반으로 작동하여 대량의 문서를 처리하기에 시간이 과도하게 소요됨.
  - **테이블 인식 실패**: 육안으로 명확히 테이블로 보이는 영역도 레이아웃 분석 과정에서 **일반 텍스트(Plain Text)**로 처리해버리는 경우가 빈번함. (False Negative)

### 3. `mineru`
- **Status**: 보류
- **결과**:
  - **정확한 레이아웃 분석**: 테이블 및 문서 구조(Layout) 인식 자체는 매우 정확하게 작동함.
  - **한글 OCR 품질 저하**: 한글 텍스트 인식 과정에서 심각한 오류가 발생함.
    - **이상한 문자 혼입**: 예) `기타fos科学发展道路上(개선)로uman` 와 같이 한글, 중국어, 영어가 뒤섞인 깨진 텍스트가 생성됨.
    - **반복 생성**: 특정 단어가 비정상적으로 반복되는 현상(Hallucination)이 관찰됨.

## Experiment 2: Table Header Identification Strategy

테이블 추출 시 가장 큰 문제 중 하나인 **'헤더(Header) 영역의 모호성'**을 해결하기 위해 새로운 접근 방식을 도입하였습니다.

### Color-based Header Detection (Row Background Color)
- **접근 방식**: PDF 문서 내부의 그래픽 요소(Rect) 분석을 통해 테이블 각 행(Row)의 배경색을 추출.
- **가설**: "일반적으로 테이블의 헤더 영역은 본문(Body)과 시각적으로 구분하기 위해 **특정 배경색(Shading)**이 적용되어 있다."
- **구현 결과**:
  - **헤더 자동 식별**: 배경색이 존재하는 상단 연속 행을 헤더로 자동 인식하여 일반 데이터 행과 구분 성공.
  - **테이블 구조 보존**: Markdown 변환 시, 인식된 식별된 헤더 영역과 바디 영역 사이에만 구분선(`|---|---|`)을 삽입하여, 여러 줄로 구성된 복잡한 헤더도 원형 그대로 시각화 가능.
  

## Experiment 3: Logical Table Merging Strategy (Multi-page & Split)

단순히 페이지 별로 테이블을 추출할 경우, 하나의 테이블이 페이지 넘김으로 인해 절단되면 **'헤더가 중복된 두 개의 테이블'** 또는 **'헤더가 없는 본문 테이블'**로 분리되는 문제가 발생합니다. 이를 해결하기 위해 논리적 병합 로직을 고도화했습니다.


### Table Filtering (New)
- **Overlap Filtering**: 큰 테이블 안에 포함된 작은 테이블(중복 인식된 영역)을 사전에 제거하여 데이터 중복 및 병합 오류를 방지합니다.

### Merging Logic (Priority Based)
1.  **Body Continuation (Priority 1 - Force Merge)**:
    - **조건**: 이어지는 테이블(`Table 2`)의 첫 번째 행에 **배경색이 없는 경우(`None`)**.
    - **동작**: 헤더가 생략된 본문의 연속으로 판단하여, 무조건 이전 테이블에 **병합(Append)**합니다. (컬럼 개수 불일치 시에도 사용자 규칙 우선 적용)
2.  **Header Continuation (Priority 2 - Split Header)**:
    - **조건**: 이전 테이블이 **유색 행(Color)**으로 끝나고, 다음 테이블이 **유색 행(Color)**으로 시작하는 경우.
    - **동작**: 페이지 넘김 등으로 인해 헤더 자체가 분할된 것으로 판단하여, 다음 테이블을 바로 **병합(Append)**합니다.

### Result
이러한 로직을 통해 복잡한 표가 여러 페이지에 걸쳐 있거나 헤더가 잘린 경우에도 **문서 전체에서 하나의 논리적 테이블**로 완벽하게 재구성할 수 있게 되었습니다.

## Next Steps
- `pdfplumber`를 기반으로 한 '공백 컬럼 제거' 및 '헤더 기반 구조화' 로직으로 전처리 파이프라인 확정.
- 추가적인 실험(Experiment 4)을 통해 대형 테이블의 청킹 및 배치 처리 프로세스 구축 완료 (하단 참조).

## Experiment 4: Table Separation & Chunking (New)

RAG 시스템의 성능을 최적화하기 위해 **대형 테이블 처리**와 **텍스트-테이블 분리** 파이프라인을 구축했습니다.

### 1. Table Separation (Placeholders)
- **개념**: 문서의 텍스트 흐름에서 테이블을 분리하여 별도로 저장하고, 본문에는 `[TABLE_ID]` 형태의 플레이스홀더를 심습니다.
- **구조화된 저장**: 추출된 테이블은 마크다운 문자열뿐만 아니라, **헤더(Header)**와 **본문(Body)**이 분리된 JSON 객체로 저장되어, 후속 처리(LLM 입력 등)에서 유연하게 활용 가능합니다.

### 2. Table Chunking (Context Preservation)
- **Problem**: 토큰 제한(Context Window)을 초과하는 대형 테이블을 그대로 입력하면 일부가 잘리거나 비용이 과다하게 발생합니다.
- **Solution (`tiktoken` based)**:
    - **토큰 계산**: `gpt-4o` 인코딩(`o200k_base`)을 기준으로 테이블 본문의 토큰 수를 계산합니다.
    - **스마트 분할**: 설정된 `MAX_TOKENS`(예: 1024)를 초과할 경우, 본문을 여러 청크로 나눕니다.
    - **헤더 보존 (Key Feature)**: 분할된 각 청크(`TABLE_N_0`, `TABLE_N_1`...)마다 **원본 헤더를 복제하여 포함**시킵니다. 이를 통해 모델은 데이터가 어떤 컬럼에 해당하는지 문맥을 잃지 않고 이해할 수 있습니다.

### 3. Batch Processing
- **기능**: 단일 파일뿐만 아니라, 특정 디렉토리 내의 모든 PDF 파일을 일괄 처리하는 파이프라인을 구현했습니다.
- **출력**: 각 파일별로 전용 결과 폴더를 생성하여 텍스트 파일과 JSON 데이터를 체계적으로 저장합니다.

## Experiment 5: Table-Aware Chunking & Batch API Workflow (New)

대규모 문서 처리를 위한 **OpenAI Batch API** 연동 파이프라인과 **테이블 문맥 보존(Table-Aware Chunking)** 전략을 고도화했습니다.

### 1. Table-Aware Text Splitting (Recursive Expansion)
- **Problem**: 텍스트 청킹 과정에서 `[TABLE_0_2]`(Table 0의 0~2번 청크 로드 필요)와 같은 플레이스홀더가 포함될 경우, 단순 텍스트 분할 시 테이블의 일부만 포함되거나 문맥이 끊길 위험이 있습니다.
- **Solution (`TableAwareSplitter`)**:
    - **Recursive Splitting**: 먼저 `RecursiveCharacterTextSplitter`로 텍스트를 적절한 크기(예: 4000자)로 나눕니다.
    - **Placeholder Expansion**: 청크 내에서 테이블 플레이스홀더(`[TABLE_ID]`)가 발견되면, 해당 테이블의 실제 Markdown 내용을 주입하여 **Context를 완성**합니다.
    - **Single Table Constraint**: 하나의 텍스트 청크에는 **최대 1개의 테이블**만 포함되도록 토큰 비용(Penalty)을 부여하여, LLM이 한 번에 하나의 복잡한 테이블에만 집중할 수 있도록 강제했습니다.

### 2. Batch Processing Pipeline (3-Stage)
비용 절감(50%)과 대량 처리를 위해 OpenAI Batch API를 활용하는 3단계 파이프라인을 구축했습니다.

#### Step 1: Preparation (`prepare_batch.py`)
- **기능**: 처리된 `final_chunks.json`을 읽어 OpenAI Batch API 입력 포맷(`.jsonl`)으로 변환.
- **Improved Schema Enforcement (v2)**: 단순 프롬프트 대신 **OpenAI Tools (Function Calling)**를 사용하여 `GraphExtraction` 스키마 100% 준수를 보장. Markdown 코드 블록이 아닌 순수 JSON 추출.
- **Table Injection**: `[TABLE_ID]`를 실제 Markdown Table로 교체하여 Context 완성.

#### Step 2: Submission (`submit_batch.py`)
- **기능**: `.jsonl` 파일을 업로드하고 Batch Job 생성.
- **Selective Submission**: 명령줄 인자로 **특정 파일만 선택 전송** 가능 (예: `python -m src...submit_batch file1.jsonl`).
- **Tracking**: `submitted_jobs.json`에 ID 및 파일명 기록.

#### Step 3: Monitoring & Processing (`check_batch_status.py`, `process_batch_results.py`)
- **Status Check**: `check_batch_status.py`를 통해 모든 작업의 진행 상황을 테이블 형태로 한눈에 확인.
- **Result Processing**: 작업 완료 시 결과를 다운로드/파싱. 재시도(Retry) 로직 포함.

#### Step 4: Graph Construction (`create_graph_nx.py`)
- **기능**: 로컬에 다운로드된 완료 결과(`batch_output_*.jsonl`)를 취합하여 **NetworkX 그래프**(`graph.gexf`, `graph.graphml`) 생성.
- **Robust Parsing**: Tool Calls 포맷과 레거시 포맷 모두 지원, `None` 속성 자동 처리를 통한 에러 방지.

### 3. Error Handling: Split & Retry Strategy
Batch 작업 중 **'실패(Failed)'** 또는 **'토큰 한도 초과(Token Limit Exceeded)'** 오류가 발생할 경우를 대비해 자동 복구 로직을 구현했습니다.

- **Trigger**: 작업 상태가 `failed`, `expired`, `cancelled`인 경우.
- **Action**:
    1. 실패한 작업의 **원본 입력 파일(Input File)**을 다운로드.
    2. 파일 내용을 **정확히 절반(Half)**으로 분할 (`Part A`, `Part B`).
    3. 2개의 새로운 Batch Job으로 **재전송(Resubmit)**.
- **Effect**: 한 번에 너무 많은 요청이 몰려 실패하는 경우, 작업을 더 작은 단위로 쪼개어 성공률을 높입니다.


## Experiment 6: Prompt Engineering for Graph Structure Optimization (New)

지식 그래프의 품질을 저해하는 3대 문제(고립 노드, 성게 현상, 노드 충돌)를 해결하기 위해 **프롬프트 엔지니어링**을 통한 구조적 제약을 도입했습니다.

### 1. 고립된 조직 (Isolates) 해결
- **Problem**: 'MX사업부', 'DS부문' 등 하위 조직이 '삼성전자' 본사와 연결되지 않아, 검색 시 정보가 단절되는 현상.
- **Solution (Implicit Subject Enforcement)**:
    - 프롬프트에 **"이 문서는 삼성전자의 보고서이다"** 라는 전제를 명시.
    - 모든 추출된 하위 조직(Division)이나 투자 대상 기업이 **반드시 '삼성전자'와 `HAS_DIVISION` 또는 `INVESTED_IN` 관계로 연결되도록 강제**.
    - **Effect**: 그래프의 연결성(Connectivity)이 대폭 향상되어 '거대 컴포넌트(Giant Component)' 비율이 상승함.

### 2. 날짜의 노드화 (Date as Node) 방지 - "성게 현상" 해결
- **Problem**: '2024년 1월', '2023년' 등이 별도의 노드로 생성되면서, 수많은 엔티티가 이 날짜 노드 하나에 몰리는 **성게(Sea Urchin) 모양의 클러스터** 발생. 이는 그래프 탐색 효율을 떨어뜨림.
- **Solution (Event-Centric Extraction)**:
    - **Rule**: "날짜(Date)는 노드로 만들지 말고, **관계(Edge)의 설명(Description)이나 속성**으로 내린다."
    - **Exceptions**: '갤럭시 언팩'과 같이 고유한 **사건(Event)** 명칭만 노드로 추출.
    - **Effect**: 그래프가 의미 없는 날짜 허브로 인해 복잡해지는 것을 방지하고, 정보의 맥락이 엣지에 보존됨.

### 3. 수치의 노드화 (Metric as Node) 방지 - "노드 충돌" 해결
- **Problem**: '77.8%', '100억원' 같은 수치가 노드가 될 경우, 서로 다른 맥락(매출 증가율 vs 점유율)에서 같은 숫자가 나오면 **잘못된 연결(Collision)**이 발생함.
- **Solution (Metric to Edge Property)**:
    - **Rule**: **"분석 목적이 아니라면 숫자는 노드로 생성 금지."**
    - 숫자는 엔티티 간의 관계를 설명하는 **텍스트(Description)**에 포함시킴.
    - 예: `(삼성전자) -> [RECORDED_REVENUE] -> (HBM 시장)` 엣지의 설명에 "2024년 3분기 매출 10조원 달성"이라고 기록.
    - **Effect**: 그래프의 의미적 정확도가 향상되고, 불필요한 노드 생성을 억제함.

### 4. 띄어쓰기 및 OCR 오류 (OCR Error) 해결
- **Problem**: '삼 성 웰 스 토 리', '삼성 전자 로지 텍'과 같이 PDF 추출 과정에서 글자 사이에 공백이 삽입되거나, 표의 세로선이 문자를 가르는 현상.
- **Solution (Mental Reconstruction)**:
    - **Rule**: "텍스트를 추출하기 전에 **정신적 복구(Mental Reconstruction)** 과정을 거쳐라."
    - 구체적인 예시 명시: "삼 성 웰 스 토 리 (주)" -> "삼성웰스토리(주)".
    - **Effect**: 엔티티명 중복(삼성웰스토리 vs 삼 성 웰 스 토 리)을 방지하고 정확한 노드 병합 유도.

### 5. 불필요한 노드 (Noise) 차단
- **Problem**: '기타', '합계', '상장주식', '총계' 등 일반 명사나 집계용 단어가 조직(Organization) 노드로 잘못 추출됨.
- **Solution (Blocklist)**:
    - **Rule**: "일반 명사나 집계용 단어는 **노드 생성 금지 목록(Blocklist)**에 포함하여 절대 추출하지 말 것."
    - 금지어: '시장', '기술', '변화', '미래', '기타', '계', '합계', '총계', 'Others', 'Total'.
    - **Effect**: 그래프의 품질(S/N Ratio)을 높이고, 무의미한 허브 생성 방지.

### 6. 값 매핑 오류 (Table Alignment) 해결
- **Problem**: 테이블의 '합계(Total)' 행에 있는 수치(전체 매출 등)를 리스트의 첫 번째 회사나 엉뚱한 회사에 매핑하는 환각(Hallucination) 발생.
- **Solution (Explicit Mapping Rule)**:
    - **Rule**: "테이블의 **'Total' 또는 '계' 행은 무시**하고, 개별 항목(Individual Items)만 연결하라."
    - 헤더와 값의 수직 매핑(Vertical Alignment)을 재확인하도록 지시.
    - **Effect**: 재무 데이터나 통계 수치의 관계 추출 정확도 개선.
    - **Effect**: 재무 데이터나 통계 수치의 관계 추출 정확도 개선.

## Experiment 7: Quantitative Diagnosis of Graph Health (Results)

프롬프트 엔지니어링(Experiment 6) 적용 전후의 그래프 건강 상태를 정량적으로 비교 분석했습니다.

### Metric Comparison

| Metric | Before (Initial Prompt) | After (Refined Prompt) | Improvement |
| :--- | :--- | :--- | :--- |
| **Nodes** | 9,500 | **5,034** | **-47%** (노이즈/중복 제거) |
| **Edges** | 11,982 | 6,167 | -48% (무의미한 연결 감소) |
| **Components** | 644 | **103** | **-84%** (연결성 대폭 강화) |
| **Giant Component** | 92.99% | **97.52%** | **+4.53%p** (거의 완전한 연결) |
| **Isolates (고립)** | 629 (6.62%) | **90 (1.79%)** | **-71%** (고립 문제 해결) |
| **Leaves (잔가지)** | 6,692 (70.44%) | 3,908 (77.63%) | 절대 수 감소, 구조적 정리됨 |

### Analysis
1.  **Noise Reduction**: 노드 수가 47% 감소했습니다. 이는 날짜(Date), 단순 수치(Metric), 불용어(Stopwords) 등이 노드에서 제거되고 엣지 속성으로 올바르게 흡수되었음을 의미합니다.
2.  **Connectivity Surge**: 컴포넌트(섬)의 개수가 644개에서 103개로 급감하고, 거대 컴포넌트 비율이 97.5%에 도달했습니다. 이는 `Implicit Subject Enforcement` 규칙이 매우 효과적으로 작동했음을 보여줍니다.
3.  **High-Quality Graph**: 고립 노드가 1.79%에 불과하여, 생성된 지식 그래프의 정보 도달 가능성(Accessibility)이 매우 높아졌습니다.

### Conclusion
**"Refined Prompt"**는 단순한 텍스트 추출을 넘어, 그래프의 **토폴로지(Topology)를 최적화**하는 데 결정적인 역할을 수행했습니다. 이제 이 고품질 그래프를 기반으로 커뮤니티 탐지 및 서머리를 진행합니다.
