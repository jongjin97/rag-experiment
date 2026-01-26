import os
import asyncio
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import MODEL_NAME
from src.prompts.prompt import GRAPH_RAG_KOR_EXTRACTOR_PROMPT

# --- Prompts ---
# Self-Reflection Check Prompt
# Boolean decision to determine if we missed entities. 
GLEANING_CHECK_PROMPT = """
당신은 지식 그래프 추출 작업의 품질을 검수하는 관리자(Supervisor)입니다.
아래 제공된 [Context]와 이미 추출된 [Extracted Entities]를 비교하여, **중요한 엔티티나 관계가 누락되었는지** 판단하십시오.

[Context] 내용을 완벽하게 커버하기 위해 추가적인 추출이 필요합니까?
'네' 또는 '아니오'로 판단하십시오.

- 이미 핵심 정보가 충분히 추출되었다면 -> False (중단)
- 중요한 인물, 조직, 사건, 또는 세부적인 관계가 누락되었다면 -> True (추가 추출 필요)

[Context]:
{context}

[Extracted Entities]:
{history}
"""

# Gleaning Extraction Prompt
# Continuation prompt assuming "MANY entities were missed" logic from the paper.
GLEANING_EXTRACTION_PROMPT = """
당신은 지식 그래프 추출 전문가입니다.
앞선 검토 결과, **기존 추출 목록에 다수의 중요한 엔티티와 관계가 누락되었음**이 확인되었습니다.

기존 목록과 중복되지 않는, **새로운 엔티티와 관계**를 [Context]에서 추가로 추출하십시오.
특히, 문맥의 세부사항(수치, 날짜, 구체적인 행위)을 놓치지 않도록 주의하십시오.

[Context]:
{context}

[Existing Lists (Do NOT duplicate these precise pairs)]:
{history}
"""

# --- Data Structures ---
# Re-using structures from extractor.py but defining here to be self-contained or we can import.
# For gleaning, we need the exact same structure to merge lists easily.

class Entity(BaseModel):
    """지식 그래프의 노드(Node) 정보를 담습니다."""
    name: str = Field(
        description="엔티티의 고유 이름. 반드시 본문의 '한국어 원문'을 그대로 사용할 것. (영어 번역 금지, 중복 제거)"
    )
    type: str = Field(
        description="엔티티의 타입. [ORGANIZATION, PRODUCT, CONCEPT, GEO, EVENT, METRIC] 중 하나."
    )
    description: str = Field(
        description="엔티티에 대한 15~30자 내외의 한국어 요약 설명."
    )

class Relationship(BaseModel):
    """두 엔티티 간의 관계(Edge) 정보를 담습니다."""
    source: str = Field(
        description="관계의 주체(출발 노드) 이름. Entity.name과 일치해야 함."
    )
    target: str = Field(
        description="관계의 대상(도착 노드) 이름. Entity.name과 일치해야 함."
    )
    relation_type: str = Field(
        description="관계의 성격을 나타내는 영어 대문자 동사 (예: RECORDED, PRODUCED)."
    )
    description: Optional[str] = Field(
        description="관계에 대한 구체적인 서술 (수치, 기간 등 포함)."
    )

class GraphExtraction(BaseModel):
    """추출된 지식 그래프 데이터셋."""
    entities: List[Entity]
    relationships: List[Relationship]

class GleaningDecision(BaseModel):
    needs_more_extraction: bool = Field(
        description="추가적인 엔티티 추출이 필요한지 여부. True면 추가 추출 진행, False면 종료."
    )

# --- Class Implementation ---

class GleaningGraphExtractor:
    def __init__(self, model_name: str = MODEL_NAME, max_gleanings: int = 3):
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0.0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.max_gleanings = max_gleanings
        
        # 1. Main Extractor
        if hasattr(self.llm, "with_structured_output"):
            self.extractor_runnable = self.llm.with_structured_output(GraphExtraction)
            self.decision_runnable = self.llm.with_structured_output(GleaningDecision)
        else:
            # Fallback for older LangChain versions (though unlikely in this env)
            self.extractor_runnable = self.llm | PydanticOutputParser(pydantic_object=GraphExtraction)
            self.decision_runnable = self.llm | PydanticOutputParser(pydantic_object=GleaningDecision)
            
        self.initial_system_prompt = GRAPH_RAG_KOR_EXTRACTOR_PROMPT
        self.check_system_prompt = GLEANING_CHECK_PROMPT
        self.glean_system_prompt = GLEANING_EXTRACTION_PROMPT

    def _format_history(self, extraction: GraphExtraction) -> str:
        """Formats the current extraction into a string for the prompt."""
        entities_str = ", ".join([e.name for e in extraction.entities])
        return f"Entities: [{entities_str}]"

    async def extract_async(self, text_chunk: str) -> GraphExtraction:
        """
        Extracts entities/relationships with a Self-Reflection Loop (Gleaning).
        """
        # Step 1: Initial Extraction
        print("Running Initial Extraction...")
        final_extraction = await self._run_initial_extraction(text_chunk)
        
        # Step 2: Gleaning Loop
        for i in range(self.max_gleanings):
            # Check if we need more
            decision = await self._check_completeness(text_chunk, final_extraction)
            
            if not decision.needs_more_extraction:
                print(f"Gleaning Loop {i+1}: Completeness Check -> Satisfied.")
                break
            
            print(f"Gleaning Loop {i+1}: Detected missing entities. Gleaning...")
            
            # Extract more
            new_extraction = await self._run_gleaning_extraction(text_chunk, final_extraction)
            
            # Merge
            pre_count = len(final_extraction.entities)
            final_extraction = self._merge_extractions(final_extraction, new_extraction)
            post_count = len(final_extraction.entities)
            
            print(f"  -> Added {post_count - pre_count} new entities.")
            
            if post_count == pre_count:
                print("  -> No new unique entities found. Stopping loop.")
                break
                
        return final_extraction

    async def _run_initial_extraction(self, text: str) -> GraphExtraction:
        messages = [
            ("system", self.initial_system_prompt),
            ("human", f"Context:\n{text}")
        ]
        try:
            return await self.extractor_runnable.ainvoke(messages)
        except Exception as e:
            print(f"Initial Extraction Error: {e}")
            return GraphExtraction(entities=[], relationships=[])

    async def _check_completeness(self, text: str, current_extraction: GraphExtraction) -> GleaningDecision:
        history_str = self._format_history(current_extraction)
        messages = [
            ("human", self.check_system_prompt.format(context=text, history=history_str))
        ]
        try:
            return await self.decision_runnable.ainvoke(messages)
        except Exception as e:
            print(f"Completeness Check Error: {e}")
            return GleaningDecision(needs_more_extraction=False)

    async def _run_gleaning_extraction(self, text: str, current_extraction: GraphExtraction) -> GraphExtraction:
        history_str = self._format_history(current_extraction)
        
        # We reuse the same output schema (GraphExtraction) but with a different prompt
        # We construct the messages manually for the gleaning prompt logic
        messages = [
            # We can treat this as a system instruction or a very strong human instruction
            ("system", self.glean_system_prompt.format(context=text, history=history_str)),
            ("human", "누락된 엔티티와 관계를 추가로 추출해줘.")
        ]
        try:
            return await self.extractor_runnable.ainvoke(messages)
        except Exception as e:
            print(f"Gleaning Extraction Error: {e}")
            return GraphExtraction(entities=[], relationships=[])

    def _merge_extractions(self, base: GraphExtraction, new: GraphExtraction) -> GraphExtraction:
        """Merges 'new' into 'base', avoiding duplicates by name."""
        existing_names = {e.name for e in base.entities}
        existing_relations = {(r.source, r.target, r.relation_type) for r in base.relationships}
        
        merged_entities = list(base.entities)
        merged_relations = list(base.relationships)
        
        for e in new.entities:
            if e.name not in existing_names:
                merged_entities.append(e)
                existing_names.add(e.name)
        
        for r in new.relationships:
            key = (r.source, r.target, r.relation_type)
            if key not in existing_relations:
                merged_relations.append(r)
                existing_relations.add(key)
                
        return GraphExtraction(entities=merged_entities, relationships=merged_relations)

# --- Test Block ---
if __name__ == "__main__":
    async def main():
        sample_text = """
        삼성전자는 2024년 1월 17일 미국 새너제이에서 '갤럭시 언팩 2024' 행사를 열고 갤럭시 S24 시리즈를 공개했다.
        이번 시리즈는 온디바이스 AI를 탑재하여 실시간 통역 기능을 제공한다.
        노태문 MX사업부장(사장)은 "갤럭시 S24 시리즈는 모바일 AI의 새로운 시대를 열 것"이라고 강조했다.
        한편, 경쟁사인 애플은 아이폰 16 출시를 준비 중이며, TSMC와의 협력을 강화하고 있다.
        SK하이닉스 또한 HBM3E 시장에서의 주도권을 쥐기 위해 공격적인 투자를 단행했다.
        """
        
        extractor = GleaningGraphExtractor(max_gleanings=2)
        print("Start Extraction...")
        result = await extractor.extract_async(sample_text)
        
        print(f"\nFinal Entity Count: {len(result.entities)}")
        print("Entities:")
        for e in result.entities:
            print(f" - {e.name} ({e.type})")
            
        print("\nRelationships:")
        for r in result.relationships:
            print(f" - {r.source} -> {r.relation_type} -> {r.target}")

    asyncio.run(main())
