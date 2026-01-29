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
- **Table Injection**: 텍스트 내의 `[TABLE_ID]` 플레이스홀더를 `extracted_tables.json`에 저장된 **실제 Markdown Table** 원본으로 교체.
- **Token Management**: `tiktoken`을 사용하여 각 배치 파일이 **1.5M 토큰**을 넘지 않도록 자동 분할 저장 (`batch_input_part_N.jsonl`).

#### Step 2: Submission (`submit_batch.py`)
- **기능**: 생성된 `.jsonl` 파일을 OpenAI 서버에 업로드하고 Batch Job을 생성.
- **Tracking**: 작업 ID와 상태를 `submitted_jobs.json`에 기록하여 추적 관리.

#### Step 3: Result Processing (`process_batch_results.py`)
- **기능**: 제출된 작업의 상태를 확인(Polling)하고, 완료 시 결과를 다운로드하여 파싱.
- **출력**: 최종적으로 그래프 추출 결과(Entity, Relationship)를 저장.

