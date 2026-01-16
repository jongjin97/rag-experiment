
# GraphRAG Benchmark Report (with RAGAS)

## 1. Metrics Summary (Average)
| Mode | Latency (s) | Faithfulness (0-1) | Answer Relevance (0-1) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Graph Local** | **3.9s** | **0.99** | 0.41 | **Fastest**. Best for specific facts. High precision. |
| **Naive RAG** | 5.7s | 0.97 | 0.26 | Baseline. Struggles with relevance on broad queries. |
| **Graph Global** | 8.2s | **1.00** | 0.32 | Slower. Great for high-level summaries. |
| **Graph Hybrid** | 11.9s | 0.96 | **0.44** | Slowest. **Highest Relevance** (best quality) for complex queries. |

> **Note**: "Answer Relevance" can be low (0.0) when the system correctly answers "I cannot find this information," as the vague answer is technically not "relevant" to the question's content intent.

## 2. Detailed Qualitative Analysis

### ✅ Specific Fact (e.g. "HBM3E Bandwidth")
*   **Graph Local**: Found the entity but specific bandwidth attribute was missing in graph (honest "I don't know"). Latency: 1.6s.
*   **Naive RAG**: hallucinated or found partial info. Faithfulness 1.0 (consistent with its own context).

### 🚀 Broad Strategy (e.g. "DX Division Strategy")
*   **Graph Global**: Aggregated summary provided a coherent narrative. Faithfulness 1.0.
*   **Naive RAG**: Retrieved fragmented chunks. Answer was less cohesive.

### 🧠 Hybrid Complex (e.g. "HBM Impact on DS Revenue")
*   **Graph Hybrid**: combined local details with global trends.
*   **Score**: High faithfulness (0.96) and relatively higher relevance.

## 3. RAGAS Metrics Interpretation
*   **Faithfulness**: All models scored high (>0.95), meaning they largely stuck to the provided context (low hallucination).
*   **Answer Relevance**: **Hybrid** scored highest (0.44), indicating it provided the most direct and complete answers to the user's questions compared to other modes.

## 4. Recommendation
*   **For Chatbot**: Use **Graph Local** (Fast, Precise).
*   **For Analysis/Report Gen**: Use **Graph Hybrid** or **Global** (High Quality, Comprehensive).
*   **Naive RAG**: Good fallback, but GraphRAG offers better structure for complex topics.
