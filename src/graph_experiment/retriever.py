
import networkx as nx
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import MODEL_NAME, DATA_DIR

# Load environment
load_dotenv()

from src.graph_experiment.extractor import GraphExtraction

QUERY_ENTITY_PROMPT = """
You are a search assistant. Identify the key entities in the user's query that we should look up in the knowledge graph.
- Extract Companies, People, Products, Events, or specific concepts.
- Ignore generic words like "what", "where", "influence", "effect".
- Return the result in the 'entities' field of the JSON.
- The 'relationships' field can be empty.
"""

GRAPH_FILE = DATA_DIR / "processed_experiment" / "batch_input2" / "graph.gexf"

class LocalSearchRetriever:
    def __init__(self, graph_path: Path = GRAPH_FILE):
        self.graph_path = graph_path
        print(f"Loading Graph from {self.graph_path}...")
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found at {self.graph_path}")
            
        self.G = nx.read_gexf(self.graph_path)
        print(f"Graph Loaded: {self.G.number_of_nodes()} Nodes, {self.G.number_of_edges()} Edges.")
        
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
        
        # Query Extractor
        if hasattr(self.llm, "with_structured_output"):
            self.query_extractor = self.llm.with_structured_output(GraphExtraction)
        else:
            # Fallback (though gpt-4o supports it)
            from langchain_core.output_parsers import PydanticOutputParser
            self.query_extractor = self.llm | PydanticOutputParser(pydantic_object=GraphExtraction)

    def find_entry_points(self, query: str):
        """Find matching nodes using LLM-extracted entities."""
        print(f"Extracting entities from query: '{query}'...")
        
        try:
            messages = [
                ("system", QUERY_ENTITY_PROMPT),
                ("user", query)
            ]
            result = self.query_extractor.invoke(messages)
            extracted_names = [e.name for e in result.entities]
            print(f" -> Extracted Candidates: {extracted_names}")
        except Exception as e:
            print(f"Error in query extraction: {e}")
            extracted_names = query.split() # Fallback

        matches = []
        # Match extracted names against Graph Nodes (Fuzzy/Substring)
        # Use simple substring check or exact match preference
        for target in extracted_names:
            target_lower = target.lower().replace(" ", "")
            
            # 1. Exact Match (Best)
            if target in self.G.nodes():
                matches.append(target)
                continue
                
            # 2. Substring Match
            found_sub = False
            for node in self.G.nodes():
                node_str = str(node)
                # Check if target matches node or node matches target (flexible)
                # e.g. target="Galaxy S24" matches node="Galaxy S24 Ultra"
                if target_lower in node_str.lower().replace(" ", ""):
                    matches.append(node)
                    found_sub = True
            
            if not found_sub:
                print(f"    - Warning: '{target}' not found in graph.")

        # Dedup
        matches = list(set(matches))
        return matches[:10]  # Limit context 

    def get_local_context(self, entry_nodes: list[str], max_tokens: int = 4000) -> str:
        """Gather 1-hop neighborhood context for entry nodes."""
        context_lines = []
        visited_edges = set()
        
        context_lines.append(f"--- Focused Entities: {', '.join(entry_nodes)} ---")
        
        for node in entry_nodes:
            if node not in self.G.nodes: continue
            
            # Node Info
            # context_lines.append(f"Entity: {node}")
            
            # 1-Hop Neighbors (Outgoing & Incoming)
            # GEXF from NetworkX is usually directed or undirected with attributes.
            # We treat it as undirected for context gathering to see all relations.
            
            neighbors = list(self.G.neighbors(node))
            
            for nbr in neighbors:
                # Get edge data
                # Handle MultiGraph or simple Graph
                edges_data = self.G.get_edge_data(node, nbr)
                
                # NetworkX get_edge_data returns a dict for simple graphs
                # or a dict of dicts for MultiGraphs.
                # Assuming simple Graph/DiGraph or taking the first edge.
                
                relation = "RELATED_TO"
                description = ""
                
                if isinstance(edges_data, dict):
                    # Check if it's a multigraph format (integer keys)
                    if 0 in edges_data: # MultiGraph
                        data = edges_data[0]
                    else:
                        data = edges_data
                    
                    relation = data.get('relation_type', 'RELATED_TO')
                    description = data.get('description', '')
                
                # Format: (Source) -> [RELATION] -> (Target) : Description
                # Normalize direction for readability
                edge_ws = f"({node}) -> [{relation}] -> ({nbr})"
                if description:
                    edge_ws += f" : {description}"
                
                if edge_ws not in visited_edges:
                    context_lines.append(edge_ws)
                    visited_edges.add(edge_ws)
        
        # Simple Token Limit Truncation (Estimate 1 line ~ 20 tokens)
        print("\n".join(context_lines)[:max_tokens * 4])
        return "\n".join(context_lines)[:max_tokens * 4]

    def answer_question(self, query: str):
        print(f"\nQuery: {query}")
        
        # 1. Find Entry Points
        entry_points = self.find_entry_points(query)
        if not entry_points:
            return "No relevant entities found in the graph to answer this question."
        
        print(f"Found Entry Points: {entry_points}")
        
        # 2. Build Context
        context = self.get_local_context(entry_points)
        print(f"Context Length: {len(context)} chars")
        # print(f"Context Preview:\n{context[:500]}...")
        
        # 3. Generate Answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering based ONLY on the provided Knowledge Graph Context.
            
            Context:
            {context}
            
            - If the answer is found in the context, synthesize it clearly.
            - If the context doesn't contain the answer, say "I don't have enough information."
            - Cite specific entities or relations if possible.
            - Answer in Korean.
            """),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})
        
        return response

def main():
    retriever = LocalSearchRetriever()
    
    while True:
        try:
            q = input("\n질문 입력 (종료: q): ")
            if q.lower() in ['q', 'quit', 'l']:
                break
            
            answer = retriever.answer_question(q)
            print("\n[답변]")
            print(answer)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
