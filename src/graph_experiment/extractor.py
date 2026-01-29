import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import MODEL_NAME
from src.prompts.prompt import GRAPH_RAG_HYBRID_PROMPT
from src.config import DATA_DIR
# --- Data Structures for Graph ---

class Entity(BaseModel):
    """지식 그래프의 노드(Node) 정보를 담습니다."""
    name: str = Field(
        description="엔티티의 고유 이름. 반드시 본문의 '한국어 원문'을 그대로 사용할 것. (영어 번역 금지, 중복 제거)"
    )
    type: str = Field(
        description="엔티티의 타입. 반드시 다음 중 하나 선택: [ORGANIZATION, PRODUCT, CONCEPT, GEO, EVENT, METRIC]"
    )
    description: str = Field(
        description="엔티티에 대한 15자 내외의 한국어 요약 설명."
    )

class Relationship(BaseModel):
    """두 엔티티 간의 관계(Edge) 정보를 담습니다."""
    source: str = Field(
        description="관계의 주체(출발 노드) 이름. 위에서 추출한 Entity.name과 정확히 일치해야 함."
    )
    target: str = Field(
        description="관계의 대상(도착 노드) 이름. 위에서 추출한 Entity.name과 정확히 일치해야 함."
    )
    relation_type: str = Field(
        description="관계의 성격을 나타내는 영어 대문자 동사 (예: RECORDED, PRODUCED)."
    )
    description: Optional[str] = Field(
        description="관계에 대한 구체적인 서술. **반드시 금액(수치), 날짜(기간), 출처 등의 디테일을 포함하여** 한국어 문장으로 작성할 것."
    )

class GraphExtraction(BaseModel):
    """추출된 지식 그래프 데이터셋."""
    entities: List[Entity]
    relationships: List[Relationship]

# --- Extractor Class ---

class GraphExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0.0,
            max_tokens=2000, # Safety limit to prevent infinite loops
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # Use with_structured_output for guaranteed parsing
        if hasattr(self.llm, "with_structured_output"):
            self.runnable = self.llm.with_structured_output(GraphExtraction)
        else:
            self.parser = PydanticOutputParser(pydantic_object=GraphExtraction)
            self.runnable = self.llm | self.parser

        self.system_prompt = GRAPH_RAG_HYBRID_PROMPT

    async def extract_async(self, text_chunk: str) -> GraphExtraction:
        """
        Extracts entities and relationships asynchronously.
        """
        try:
            messages = [
                ("system", self.system_prompt),
                ("human", f"Context:\n{text_chunk}")
            ]
            return await self.runnable.ainvoke(messages)
        except Exception as e:
            print(f"Extraction Error: {e}")
            return GraphExtraction(entities=[], relationships=[])

    def extract(self, text_chunk: str) -> GraphExtraction:
        """
        Extracts entities and relationships from a text chunk (Sync wrapper).
        """
        try:
            messages = [
                ("system", self.system_prompt),
                ("human", f"Context:\n{text_chunk}")
            ]
            return self.runnable.invoke(messages)
        except Exception as e:
            print(f"Extraction Error: {e}")
            return GraphExtraction(entities=[], relationships=[])

if __name__ == "__main__":
    # Test Block
    sample_text = """
    삼성전자는 2024년 1월 17일 미국 새너제이에서 '갤럭시 언팩 2024' 행사를 열고 갤럭시 S24 시리즈를 공개했다.
    이번 시리즈는 온디바이스 AI를 탑재하여 실시간 통역 기능을 제공한다.
    DS부문은 HBM 시장에서의 경쟁력 강화를 위해 연구개발 투자를 확대하였다.
    """
    
    extractor = GraphExtractor()
    result = extractor.extract(sample_text)
    
    print("Entities:")
    for e in result.entities:
        print(f" - {e.name} ({e.type}): {e.description}")
        
    print("\nRelationships:")
    for r in result.relationships:
        print(f" - {r.source} -> {r.relation_type} -> {r.target} ({r.description})")
