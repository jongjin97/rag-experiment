import networkx as nx
import os
import json
import asyncio
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from src.graph_rag_v2.extractor import GraphExtractor
from src.config import DATA_DIR, MODEL_NAME, GEMINI_MODEL_NAME, DEEPSEEK_MODEL_NAME, MAP_BATCH_SIZE, MAX_COMMUNITY_TOKENS, KNOWLEDGE_GRAPH_V2_PATH, GRAPH_RAG_V2_DIR

GRAPH_FILE = KNOWLEDGE_GRAPH_V2_PATH

# Pydantic Model for Intermediate Answer
class IntermediateResult(BaseModel):
    """각 커뮤니티 요약본의 관련성을 평가하고 중간 답변을 생성하는 모델"""
    is_relevant: bool = Field(
        description="제공된 커뮤니티 요약 내용이 사용자의 질문에 답변하는 데 실질적으로 도움이 되는 관련 정보를 포함하고 있다면 True, 아니라면 False."
    )
    score: int = Field(
        description="사용자 질문과의 관련성 점수 (0~100점). 0점은 전혀 관련 없음, 100점은 완벽한 해답을 포함함."
    )
    answer: str = Field(
        description="오직 이 커뮤니티 요약 내용에만 기반하여 도출된 중간 답변. (외부 지식 사용 금지, 관련 정보가 없다면 '정보 없음'이라고 기재)"
    )
class GraphRetriever:
    def __init__(self, graph_path: str = str(GRAPH_FILE), provider: str = "openai"):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at {graph_path}. Please run builder first.")
            
        print(f"Loading Graph from {graph_path}...")
        self.graph = nx.read_gexf(graph_path)
        
        # Use existing extractor to find entities in the query
        self.extractor = GraphExtractor()
        
        if provider == "google":
            print(f"Initializing Gemini Model ({GEMINI_MODEL_NAME})...")
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                temperature=0.0,
                api_key=os.environ.get("GOOGLE_API_KEY")
            )
            # Use same model for mapping for now
            self.map_llm = self.llm 
        elif provider == "deepseek":
            print(f"Initializing Deepseek Model ({DEEPSEEK_MODEL_NAME})...")
            self.llm = ChatOpenAI(
                model=DEEPSEEK_MODEL_NAME,
                temperature=0.0,
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY")
            )
            self.map_llm = self.llm
        else:
            self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY2"))
            # OpenAI supports structured output well
            self.map_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY2"), model_kwargs={"response_format": {"type": "json_object"}})
        
        # --- Map (Intermediate) Prompt ---
        self.map_template = """
당신은 지식 그래프의 커뮤니티 요약본(Community Summary)을 평가하는 전문 분석가입니다.
제공된 요약본과 사용자의 질문을 분석하여 다음을 수행하십시오.

1. **관련성 판단**: 요약본이 질문에 답하는 데 필요한 정보를 포함하고 있는지 판단하십시오.
2. **중간 답변 작성**: 관련이 있다면, **오직(ONLY) 제공된 요약본의 내용에만 기반하여** 상세한 중간 답변을 작성하십시오. 외부 지식을 사용하지 마십시오.
3. **점수 평가**: 질문과의 관련성을 0점에서 100점 사이의 점수로 매기십시오. (100점: 완벽한 답변 가능, 0점: 전혀 관련 없음)

Summary:
{context}

Question:
{question}

결과는 반드시 다음 키를 포함하는 JSON 형식으로 반환해야 합니다: "is_relevant", "score", "answer".
"""
        self.map_prompt = ChatPromptTemplate.from_template(self.map_template)
        # Use JsonOutputParser for structured parsing
        self.map_chain = self.map_prompt | self.map_llm | JsonOutputParser(pydantic_object=IntermediateResult)


        # --- Final Answer Prompt ---
        self.answer_template = """
당신은 지식 그래프에서 추출된 여러 중간 답변들을 종합하여 최종 결론을 도출하는 수석 연구원입니다.
아래 제공된 '중간 그래프 답변(Intermediate Graph Answers)'들을 바탕으로 사용자의 질문에 대해 논리적이고 포괄적인 답변을 작성하십시오.

[작성 가이드라인]
1. **종합 및 요약**: 파편화된 정보들을 문맥에 맞게 매끄럽게 연결하여 하나의 완성된 글로 작성하십시오.
2. **상충 해결**: 만약 중간 답변들 사이에 서로 다른 정보(예: 수치 불일치)가 있다면, 이를 숨기지 말고 차이점을 명시하십시오.
3. **근거 기반**: 제공된 맥락(Context)에 정보가 없거나 부족할 경우, 지어내지 말고 "지식 그래프 내에 관련 정보가 부족하여 답변할 수 없습니다"라고 명확히 밝히십시오.

Intermediate Graph Answers:
{context}

Question: {question}

Answer:"""
        self.answer_prompt = ChatPromptTemplate.from_template(self.answer_template)
        self.chain = self.answer_prompt | self.llm | StrOutputParser()

    def retrieve_local_context(self, query: str, depth: int = 1) -> str:
        """
        Retrieves local graph context (1-hop neighbors) for entities found in the query.
        """
        # 1. Extract entities from query
        print("Extracting query entities...")
        try:
            query_extraction = self.extractor.extract(query)
            target_entities = [e.name for e in query_extraction.entities]
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            target_entities = []
        
        if not target_entities:
            # Fallback: simple keyword matching if extraction fails or yields nothing
            print("No structured entities found, trying raw keywords...")
            # Ideally use token matching, but let's stick to what we extracted
        
        print(f"Target Entities: {target_entities}")
        
        context_lines = []
        
        # 2. Traverse Graph
        found_nodes = set()
        for entity_name in target_entities:
            # Try exact match or partial match
            matched_node = None
            if self.graph.has_node(entity_name):
                matched_node = entity_name
            else:
                # Simple loose matching
                for node in self.graph.nodes:
                    if entity_name.lower() in node.lower() or node.lower() in entity_name.lower():
                        matched_node = node
                        break
            
            if matched_node:
                found_nodes.add(matched_node)
                node_attrs = self.graph.nodes[matched_node]
                desc = node_attrs.get('description', 'No description')
                type_ = node_attrs.get('type', 'UNKNOWN')
                context_lines.append(f"ENTITY: {matched_node} ({type_}) - {desc}")
                
                # Get Neighbors (Relationships)
                # OPTIMIZATION: Limit to Top 20 neighbors to prevent context explosion
                all_neighbors = list(self.graph.neighbors(matched_node))
                if len(all_neighbors) > 20: 
                    neighbors = all_neighbors[:20]
                    context_lines.append(f"    (Truncated {len(all_neighbors) - 20} other connections...)")
                else:
                    neighbors = all_neighbors

                for neighbor in neighbors:
                    edge_data = self.graph.get_edge_data(matched_node, neighbor)
                    relation = edge_data.get('relation', 'RELATED_TO')
                    edge_desc = edge_data.get('description', '')
                    
                    context_lines.append(f"  - [{relation}] -> {neighbor}")
                    if edge_desc:
                        context_lines.append(f"    (Details: {edge_desc})")
            else:
                pass # Silent on not found to reduce noise

        if not context_lines:
            return "No relevant graph context found."
            
        return "\n".join(context_lines)

    async def _process_community_map(self, community_id: str, summary: str, query: str) -> Dict[str, Any]:
        """
        Async Map step: Process a single community summary.
        """
        # Trim summary if too long
        if len(summary) > MAX_COMMUNITY_TOKENS * 4: # roughness char estimate
            summary = summary[:MAX_COMMUNITY_TOKENS * 4] + "...(truncated)"

        try:
            # Setup specific retry or timeout logic if needed
            response = await self.map_chain.ainvoke({"context": summary, "question": query})
            response['community_id'] = community_id
            return response
        except Exception as e:
            # print(f"Error mapping community {community_id}: {e}")
            return {"is_relevant": False, "score": 0, "answer": "", "community_id": community_id}

    def retrieve_global_context(self, query: str) -> str:
        """
        Retrieves global context using Map-Reduce parallel processing.
        """
        summary_file = GRAPH_RAG_V2_DIR / "community_summaries.json"
        
        if not summary_file.exists():
            print("Community Summaries not found. Returning empty.")
            return "" 
            
        with open(summary_file, "r", encoding="utf-8") as f:
            summaries = json.load(f)
            
        print(f"Generating Map-Reduce Global Context from {len(summaries)} communities...")
        
        # Prepare Tasks
        # Filter: maybe only top 100 largest communities if there are too many?
        # For now, take all or top N
        sorted_ids = sorted(summaries.keys(), key=lambda x: int(x))
        # Limit to e.g. 50-100 to avoid extreme costs during testing?
        # Let's limit to 50 for cost/speed balance in this demo
        target_ids = sorted_ids[:50] 
        
        async def run_map_reduce():
            tasks = []
            results = []
            
            # Batch processing
            for i in tqdm(range(0, len(target_ids), MAP_BATCH_SIZE), desc="Global Search (Map Phase)"):
                batch_ids = target_ids[i : i + MAP_BATCH_SIZE]
                batch_tasks = [
                    self._process_community_map(c_id, summaries[c_id], query) 
                    for c_id in batch_ids
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
            
            return results

        # Run Async Loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we are already in an async environment (e.g. Jupyter or FastAPI)
                # This might be tricky in a plain script if not handled, but strictly 
                # for a script `asyncio.run` is safer if no loop is running.
                # Since we are likely in a script via `runpy` or `python main.py`, `asyncio.run` usually works.
                # But if `loop.is_running()` is true, using `nest_asyncio` or simply returning awaitable is needed.
                # For safety in this environment:
                import nest_asyncio
                nest_asyncio.apply()
                raw_results = loop.run_until_complete(run_map_reduce())
            else:
                 raw_results = asyncio.run(run_map_reduce())
        except RuntimeError:
             # Fallback for "loop is running" error if get_event_loop didn't detect it or other issues
             raw_results = asyncio.run(run_map_reduce())

        # Reduce Step
        relevant_results = [r for r in raw_results if r.get('score', 0) > 0 and r.get('is_relevant')]
        
        # Sort by score descending
        relevant_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"Found {len(relevant_results)} relevant communities out of {len(target_ids)} scanned.")
        
        context_lines = []
        for r in relevant_results:
            c_id = r['community_id']
            score = r['score']
            ans = r['answer']
            context_lines.append(f"[Community {c_id} | Relevance: {score}%]: {ans}")
            
        if not context_lines:
            return "No relevant communities found via Global Search."
            
        return "\n\n".join(context_lines)

    def query(self, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Answers a question using Graph Context.
        mode: 'local' (neighbors), 'global' (summaries), 'hybrid' (both).
        Returns: Dict with keys 'result' (str) and 'context' (List[str])
        """
        context = ""
        
        # 1. Local Context (Specific Entities)
        if mode in ["local", "hybrid"]:
            # Note: local search only needs to run if we find entities. 
            # If it's a pure concept question, local search might be empty.
            local_ctx = self.retrieve_local_context(question)
            if "No relevant graph context" not in local_ctx:
                context += f"\n\n--- Local Entity Context ---\n{local_ctx}"
        
        # 2. Global Context (Community Summaries via Map-Reduce)
        if mode in ["global", "hybrid"]:
            global_ctx = self.retrieve_global_context(question)
            if "No relevant communities" not in global_ctx:
                context += f"\n\n--- Global Community Context (Map-Reduce) ---\n{global_ctx}"
        
        if not context.strip():
            context = "No relevant context found in Graph."

        print("\n--- Final Graph Context Used ---\n")
        print(context[:1000] + "... (truncated)" if len(context) > 1000 else context)
        print("\n--------------------------------\n")
        
        answer = self.chain.invoke({"context": context, "question": question})
        
        return {
            "result": answer,
            "context": [context] # RAGAS expects a list of strings
        }
