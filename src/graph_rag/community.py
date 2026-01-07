import networkx as nx
import json
import os
import asyncio
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from tqdm.asyncio import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import DATA_DIR, MODEL_NAME

GRAPH_FILE = DATA_DIR / "graph_rag" / "knowledge_graph.gexf"
SUMMARY_FILE = DATA_DIR / "graph_rag" / "community_summaries.json"

class CommunitySummarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        if not GRAPH_FILE.exists():
            raise FileNotFoundError(f"Graph file not found at {GRAPH_FILE}. Run builder first.")
            
        print(f"Loading Graph from {GRAPH_FILE}...")
        self.graph = nx.read_gexf(GRAPH_FILE)
        
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0.0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Prompt for summarizing a community
        self.summary_template = """You are an expert business analyst reviewing a knowledge graph community from Samsung Business Reports.
        
Summarize the following community of entities and relationships. 
Focus on:
1. What is the central theme of this community? (e.g., "Semiconductor Market Trend", "Mobile Division Performance")
2. Key entities and their roles.
3. Important relationships and events.
4. Any financial figures or strategic decisions mentioned.

Community Data:
{community_data}

Summary:"""
        
        self.prompt = ChatPromptTemplate.from_template(self.summary_template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def get_communities(self) -> Dict[str, List[str]]:
        """Group nodes by community ID."""
        communities = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            c_id = data.get('community')
            if c_id is not None:
                communities[str(c_id)].append(node)
        return communities

    def _prepare_community_data(self, c_id: str, nodes: List[str]) -> str:
        """Format community data (nodes & edges) for the LLM."""
        lines = [f"Community ID: {c_id}"]
        
        # Add Nodes
        lines.append("\nEntities:")
        for node in nodes:
            data = self.graph.nodes[node]
            desc = data.get('description', '')
            type_ = data.get('type', 'UNKNOWN')
            lines.append(f"- {node} ({type_}): {desc}")
            
        # Add Internal Edges
        lines.append("\nRelationships:")
        subgraph = self.graph.subgraph(nodes)
        for u, v, data in subgraph.edges(data=True):
            rel = data.get('relation', 'RELATED')
            desc = data.get('description', '')
            lines.append(f"- {u} -> {rel} -> {v}: {desc}")
            
        return "\n".join(lines)

    async def summarize_community(self, c_id: str, nodes: List[str], semaphore: asyncio.Semaphore):
        """Generate summary for a single community."""
        async with semaphore:
            text_data = self._prepare_community_data(c_id, nodes)
            
            # Skip if community is too small/trivial? (Optional optimization)
            if len(nodes) < 2: 
                return c_id, "Trivial community (single node)."
                
            try:
                summary = await self.chain.ainvoke({"community_data": text_data})
                return c_id, summary
            except Exception as e:
                print(f"Error summarizing community {c_id}: {e}")
                return c_id, ""

    async def generate_all_summaries(self, max_concurrency: int = 5):
        """Generate summaries for all communities asynchronously."""
        communities = self.get_communities()
        print(f"Generating summaries for {len(communities)} communities...")
        
        if not communities:
            print("No communities found.")
            return

        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            self.summarize_community(c_id, nodes, semaphore)
            for c_id, nodes in communities.items()
        ]
        
        summaries = {}
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
            c_id, summary = await f
            if summary:
                summaries[c_id] = summary
                
        # Save to JSON
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(summaries)} summaries to {SUMMARY_FILE}")

async def main():
    summarizer = CommunitySummarizer()
    await summarizer.generate_all_summaries(max_concurrency=5)

if __name__ == "__main__":
    asyncio.run(main())
