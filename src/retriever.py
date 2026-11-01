"""
PERSON 6 — Evaluation-Driven Adaptive RAG Agent
Retriever Module with Configurable Pipelines, Execution Tracing, and Adaptive Abstention
"""

import math
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import KnowledgeBase, KBChunk

class RetrievalTrace:
    """Detailed execution trace for an adaptive retrieval query."""
    def __init__(self, query: str, strategy: str, top_k: int):
        self.query = query
        self.strategy = strategy
        self.top_k = top_k
        self.start_time = time.time()
        self.latency_ms = 0.0
        self.candidates_considered = 0
        self.top_score = 0.0
        self.confidence_score = 0.0
        self.abstained = False
        self.abstention_reason = ""
        self.retrieved_chunks = []
        self.authority_distribution = {}

    def finish(self, chunks: List[KBChunk], abstained: bool, reason: str = ""):
        self.latency_ms = round((time.time() - self.start_time) * 1000, 2)
        self.retrieved_chunks = [c.chunk_id for c in chunks]
        self.abstained = abstained
        self.abstention_reason = reason
        for c in chunks:
            self.authority_distribution[c.authority_level] = self.authority_distribution.get(c.authority_level, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "latency_ms": self.latency_ms,
            "candidates_considered": self.candidates_considered,
            "top_score": round(self.top_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "retrieved_chunks": self.retrieved_chunks,
            "authority_distribution": self.authority_distribution
        }

class AdaptiveRetriever:
    """
    Evaluation-driven retriever supporting:
    - 'lexical': BM25 keyword retrieval
    - 'dense': TF-IDF vector cosine retrieval
    - 'hybrid': Reciprocal Rank Fusion of BM25 and Dense
    - 'adaptive': Dynamically routes between BM25/Dense/Hybrid and validates confidence
    """
    def __init__(self, kb: KnowledgeBase, default_strategy: str = "adaptive", default_top_k: int = 4, abstention_threshold: float = 0.22):
        self.kb = kb
        self.default_strategy = default_strategy
        self.default_top_k = default_top_k
        self.abstention_threshold = abstention_threshold
        self.last_trace: Optional[RetrievalTrace] = None
        
        # Prepare BM25 corpus
        self.tokenized_corpus = [c.content.lower().split() for c in self.kb.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Prepare Vector Dense corpus
        self.corpus_texts = [f"{c.title} {c.section} {c.content}" for c in self.kb.chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.dense_matrix = self.vectorizer.fit_transform(self.corpus_texts)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _lexical_search(self, query: str, top_k: int) -> List[Tuple[KBChunk, float]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        max_s = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0.05:
                norm_score = float(scores[idx] / max_s)
                results.append((self.kb.chunks[idx], norm_score))
        return results

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[KBChunk, float]]:
        q_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(q_vec, self.dense_matrix).flatten()
        ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k*2]
        results = []
        for idx in ranked_indices:
            if similarities[idx] > 0.04:
                results.append((self.kb.chunks[idx], float(similarities[idx])))
        return results

    def _hybrid_rrf(self, query: str, top_k: int, rrf_k: int = 60) -> List[Tuple[KBChunk, float]]:
        bm25_res = self._lexical_search(query, top_k=top_k*2)
        dense_res = self._dense_search(query, top_k=top_k*2)
        
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, KBChunk] = {}

        for rank, (chunk, _) in enumerate(bm25_res):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, (chunk, _) in enumerate(dense_res):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        max_rrf = sorted_items[0][1] if sorted_items else 1.0
        
        results = []
        for cid, score in sorted_items[:top_k]:
            norm_score = float(score / max_rrf)
            results.append((chunk_map[cid], norm_score))
        return results

    def retrieve(self, query: str, strategy: Optional[str] = None, top_k: Optional[int] = None) -> Tuple[List[KBChunk], RetrievalTrace]:
        strategy = strategy or self.default_strategy
        top_k = top_k or self.default_top_k
        
        trace = RetrievalTrace(query=query, strategy=strategy, top_k=top_k)
        trace.candidates_considered = len(self.kb.chunks)
        
        chosen_strategy = strategy
        if strategy == "adaptive":
            # Adaptive routing:
            # - Short exact keyword/SKU/policy queries -> lexical
            # - Conceptual/conversational questions -> hybrid
            tokens = self._tokenize(query)
            has_code_or_sku = any(re.search(r"KB-|RET|REF|ORD|\d{4}", t, re.I) for t in tokens)
            if has_code_or_sku or len(tokens) <= 3:
                chosen_strategy = "lexical"
            else:
                chosen_strategy = "hybrid"

        raw_results: List[Tuple[KBChunk, float]] = []
        if chosen_strategy == "lexical":
            raw_results = self._lexical_search(query, top_k)
        elif chosen_strategy == "dense":
            raw_results = self._dense_search(query, top_k)
        else: # hybrid
            raw_results = self._hybrid_rrf(query, top_k)

        if not raw_results:
            trace.top_score = 0.0
            trace.confidence_score = 0.0
            trace.finish(chunks=[], abstained=True, reason="No matching documents found in index.")
            self.last_trace = trace
            return [], trace

        # Compute absolute semantic similarity to anchor confidence
        q_vec = self.vectorizer.transform([query])
        dense_max = float(cosine_similarity(q_vec, self.dense_matrix).max())
        
        # Check BM25 max score
        tokens = self._tokenize(query)
        bm25_max = max(self.bm25.get_scores(tokens)) if tokens else 0.0

        top_chunk = raw_results[0][0]
        authority_weight = 1.0 if top_chunk.authority_level in ["policy", "guide"] else 0.8
        
        # Absolute confidence combines dense similarity and lexical signal
        absolute_signal = max(dense_max, min(1.0, bm25_max / 5.0))
        trace.confidence_score = absolute_signal * authority_weight
        trace.top_score = absolute_signal
        
        # Abstention evaluation: if absolute signal is weak, abstain
        if trace.confidence_score < self.abstention_threshold or (dense_max < 0.12 and bm25_max < 0.5):
            trace.finish(chunks=[], abstained=True, reason=f"Low confidence ({trace.confidence_score:.3f}) below threshold {self.abstention_threshold}.")
            self.last_trace = trace
            return [], trace

        # Precedence filtering: Current policy overrides superseded; Policy > FAQ
        final_chunks: List[KBChunk] = []
        seen_topics = set()

        for chunk, score in raw_results:
            # Filter superseded versions unless specifically historical
            if "superseded" in chunk.status and "prior to 1 august 2026" not in query.lower() and "superseded" not in query.lower() and "v3" not in query.lower():
                continue
            final_chunks.append(chunk)
            if len(final_chunks) >= top_k:
                break

        if not final_chunks:
            final_chunks = [raw_results[0][0]]

        trace.finish(chunks=final_chunks, abstained=False)
        self.last_trace = trace
        return final_chunks, trace
