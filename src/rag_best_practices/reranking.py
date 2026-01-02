import torch
import numpy as np
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertLMHeadModel, BertTokenizerFast

class BGEReranker:
    """
    DLM Reranking (Cross-Encoder)
    - Uses BAAI/bge-reranker-v2-m3
    - High Accuracy, High Latency (on CPU)
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading BGE Reranker ({model_name}) on {device}...")
        self.model = CrossEncoder(model_name, device=device, tokenizer_args={"padding": True, "truncation": True, "max_length": 512})
        
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Reranks a list of documents for a given query.
        Returns top_k documents and their scores.
        """
        if not documents:
            return [], []
            
        # Create pairs [ [query, doc1], [query, doc2], ... ]
        pairs = [[query, doc] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        top_indices = sorted_indices[:top_k]
        top_docs = [documents[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        
        return top_docs, top_scores

class TILDEReranker:
    """
    TILDE Reranking (Query Likelihood)
    - Uses ielab/TILDE (BERT-based)
    - Simulates efficient scoring by summing query token logits from document representation.
    - NOTE: This model is English-only. Performance on Korean will likely be poor.
    """
    def __init__(self, model_name: str = "ielab/TILDE", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Loading TILDE Reranker ({model_name}) on {device}...")
        self.device = device
        # Use 'bert-base-uncased' tokenizer as TILDE depends on it and might not host vocab.txt itself properly
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertLMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> Tuple[List[str], List[float]]:
        if not documents:
            return [], []

        # Tokenize Query: Get Query Token IDs (exclude special tokens if needed, but for simplicity use active count)
        # TILDE sums scores of query tokens present in the vocab
        query_inputs = self.tokenizer(query, add_special_tokens=False, return_tensors="pt").to(self.device)
        query_ids = query_inputs["input_ids"][0] # 1D tensor

        scores = []
        
        # Scoring Loop (One pass per document)
        # Note: In real TILDE, document reps are pre-computed and indexed. 
        # Here we compute on-the-fly for simulation.
        with torch.no_grad():
            for doc in documents:
                # Tokenize Document
                doc_inputs = self.tokenizer(
                    doc, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**doc_inputs)
                logits = outputs.logits # (Batch, Seq, Vocab)
                
                # Max-pool logits over sequence dimension to get "Doc represents Word" score
                # This is a simplification of TILDE's expansion logic (taking max logit for each token across manual doc tokens)
                # TILDE-v2 logic: Score(Q, D) = Sum_{q in Q} max_{d in D} LogProb(q | d) ??
                # Actually ielab/TILDE is trained to predict expansion terms.
                # Standard TILDE inference:
                # 1. Expand Doc -> Top-K tokens. (Indexing)
                # 2. Score = Sum of weights of query tokens in expanded doc.
                
                # Using direct logit access (simplified likelihood):
                # We take the max logit for each vocabulary token across the document sequence
                # (Batch=1, Seq, Vocab) -> max(dim=1) -> (1, Vocab)
                token_logits, _ = torch.max(logits, dim=1)
                
                # Now select scores for query tokens
                # If a query token is not in doc (conceptually), the model predicts how likely it is to be related.
                query_token_scores = token_logits[0, query_ids]
                
                # Sum scores
                doc_score = query_token_scores.sum().item()
                scores.append(doc_score)

        # Sort
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)[::-1]
        
        top_indices = sorted_indices[:top_k]
        top_docs = [documents[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        
        return top_docs, top_scores
