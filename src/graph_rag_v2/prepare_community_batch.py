import json
import os
import networkx as nx
import tiktoken
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from src.config import DATA_DIR, MODEL_NAME

# Config
# User confirmed 1M context, but we keep a sane limit to avoid noise.
# Top 1000 nodes ~ 80k-100k tokens.
MAX_NODES_PER_COMMUNITY = 2000 
TOKENS_PER_FILE_LIMIT = 1_500_000

GRAPH_FILE = DATA_DIR / "graph_rag_v2" / "knowledge_graph.gexf"
BATCH_DIR = DATA_DIR / "graph_rag_v2" / "batch_jobs"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_TEMPLATE = """당신은 삼성전자 사업보고서(DART)의 지식 그래프 커뮤니티를 분석하는 **수석 비즈니스 애널리스트**입니다.
제공된 엔티티와 관계 데이터를 종합적으로 검토하여, 이 커뮤니티가 나타내는 비즈니스 현황을 요약 보고서 형태로 작성하십시오.

## 분석 대상 데이터 (Community Data)
{community_data}

## 작성 가이드라인
1. **핵심 테마 정의**: 이 커뮤니티 전체를 관통하는 주제(예: 'HBM 시장 경쟁력 강화 전략', 'MX사업부 1분기 실적 호조')를 파악하여 첫 문장에 명시하십시오.
2. **주요 엔티티 및 역할**: 중심이 되는 조직(부서), 핵심 제품, 또는 전략적 개념이 무엇이며 어떤 역할을 하는지 설명하십시오.
3. **정량적 데이터 포함 (필수)**: 데이터에 포함된 **구체적인 재무 수치(매출, 영업이익), 투자 금액, 날짜**를 요약문에 반드시 포함하십시오. 뭉뚱그려 설명하지 마십시오.
4. **전략적 함의**: 단순한 사실 나열을 넘어, 이 데이터가 삼성전자의 비즈니스에 어떤 의미(기회, 위기, 성장 등)를 갖는지 해석하십시오.

## 출력 형식 (한국어)
**[핵심 테마 요약]**
(이 커뮤니티의 내용을 아우르는 3~5문장의 종합 요약. 수치와 주요 사건을 포함할 것.)

**주요 인사이트:**
- **전략/재무:** (투자, 매출 등 핵심 수치 중심)
- **제품/기술:** (주요 제품 출시, 기술 개발 현황)
- **리스크/기회:** (경쟁 상황, 시장 변화 등)"""

def count_tokens(text: str) -> int:
    try:
        from tiktoken import encoding_for_model
        enc = encoding_for_model("gpt-4")
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_community_subgraph(G, nodes):
    """Sorts nodes by degree centrality within the community to pick top ones."""
    subgraph = G.subgraph(nodes)
    
    if len(nodes) <= MAX_NODES_PER_COMMUNITY:
        return nodes
        
    # Calculate degree inside the subgraph
    degrees = dict(subgraph.degree())
    # Sort by degree desc
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    # Pick top N
    top_nodes = [n for n, d in sorted_nodes[:MAX_NODES_PER_COMMUNITY]]
    return top_nodes

def format_community_text(G, c_id, nodes):
    lines = [f"Community ID: {c_id}"]
    
    # Nodes
    lines.append("\nEntities:")
    for node in nodes:
        data = G.nodes[node]
        desc = data.get('description', '') or ""
        type_ = data.get('type', 'UNKNOWN')
        lines.append(f"- {node} ({type_}): {desc}")
        
    # Internal Edges (only if both ends are in the selected nodes)
    lines.append("\nRelationships:")
    sub_G = G.subgraph(nodes)
    for u, v, data in sub_G.edges(data=True):
        rel = data.get('relation', 'RELATED')
        desc = data.get('description', '') or ""
        lines.append(f"- {u} -> {rel} -> {v}: {desc}")
        
    return "\n".join(lines)

def prepare_community_batch():
    if not GRAPH_FILE.exists():
        print("Graph file missing.")
        return

    print(f"Loading Graph from {GRAPH_FILE}...")
    G = nx.read_gexf(GRAPH_FILE)
    print(f"Loaded {G.number_of_nodes()} nodes.")

    # Group by Community
    communities = defaultdict(list)
    for node, data in G.nodes(data=True):
        c_id = data.get('community')
        if c_id is not None:
            communities[str(c_id)].append(node)
            
    print(f"Found {len(communities)} communities.")

    files_created = []
    current_requests = []
    current_tokens = 0
    file_index = 1

    sorted_community_ids = sorted(communities.keys(), key=lambda k: len(communities[k]), reverse=True)

    for c_id in tqdm(sorted_community_ids, desc="Preparing Requests"):
        all_nodes = communities[c_id]
        
        # Filter (Top N)
        selected_nodes = get_community_subgraph(G, all_nodes)
        
        # Format
        text_data = format_community_text(G, c_id, selected_nodes)
        prompt_content = SUMMARY_TEMPLATE.replace("{community_data}", text_data)
        
        tokens_est = count_tokens(prompt_content) + 200
        
        # Batch Request Body
        request_body = {
            "custom_id": f"community_{c_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
                "temperature": 0.0,
                "max_tokens": 2000
            }
        }
        
        # Check File Limit
        if current_tokens + tokens_est > TOKENS_PER_FILE_LIMIT:
            save_batch_file(current_requests, file_index)
            files_created.append(file_index)
            file_index += 1
            current_requests = []
            current_tokens = 0
            
        current_requests.append(request_body)
        current_tokens += tokens_est

    # Save Last
    if current_requests:
        save_batch_file(current_requests, file_index)
        files_created.append(file_index)

    print(f"\nSuccessfully created {len(files_created)} community batch files.")

def save_batch_file(requests, index):
    filename = BATCH_DIR / f"community_batch_part_{index}.jsonl"
    print(f"Writing {len(requests)} requests to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    prepare_community_batch()
