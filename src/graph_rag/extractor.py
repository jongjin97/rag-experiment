import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import MODEL_NAME
from src.prompts.prompt import GRAPH_RAG_EXTRACTOR_PROMPT
from src.config import DATA_DIR
# --- Data Structures for Graph ---

class Entity(BaseModel):
    """Represents a node in the Knowledge Graph."""
    name: str = Field(description="Name of the entity, capitalized and deduped.")
    type: str = Field(description="Type of the entity (e.g., ORGANIZATION, PERSON, GEO, CONCEPT, PRODUCT).")
    description: str = Field(description="Brief description of the entity based on the context.")

class Relationship(BaseModel):
    """Represents an edge between two entities."""
    source: str = Field(description="Name of the source entity.")
    target: str = Field(description="Name of the target entity.")
    relation_type: str = Field(description="Type of relationship (e.g., LOCATED_IN, PRODUCED_BY, PART_OF).")
    description: Optional[str] = Field(description="Contextual explanation of the relationship.")

class GraphExtraction(BaseModel):
    """Container for extracted graph elements."""
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

        self.system_prompt = GRAPH_RAG_EXTRACTOR_PROMPT

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
