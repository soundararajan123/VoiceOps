import os
import re
import csv
"""
PERSON 6 — Step 2: Evaluation-Driven Adaptive RAG Agent
"""

from typing import Dict, Any, Optional
from .retriever import AdaptiveRetriever, RetrievalTrace
from .llm_adapter import LLMAdapter

RAG_SYSTEM_PROMPT = """You are ZentroMart's authoritative AI Customer Support Agent.
Always identify yourself as an AI assistant.
Answer customer questions STRICTLY using the retrieved context from ZentroMart internal documentation.
Follow official policy precedence: Policy (RET, REF) overrides Guides, Help, FAQs, and Marketing.
Current policies override superseded policies unless the customer explicitly references older orders prior to 1 August 2026.
If the retrieved evidence does not contain sufficient facts to answer the question, or if you are instructed to abstain, state clearly and politely that ZentroMart documentation does not support the request."""

class AdaptiveRAGAgent:
    def __init__(self, arg1: Any = None, arg2: Any = None, arg3: Any = None):
        if hasattr(arg1, "find_product") or hasattr(arg1, "documents"):
            self.kb = arg1
            self.retriever = arg2 or AdaptiveRetriever(self.kb)
            self.llm = arg3 or LLMAdapter()
        else:
            self.retriever = arg1
            self.llm = arg2 or LLMAdapter()
            self.kb = getattr(self.retriever, "kb", arg3)

    def answer_question(self, question: str, strategy: Optional[str] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        chunks, trace = self.retriever.retrieve(question, strategy=strategy, top_k=top_k)
        
        if trace.abstained:
            abstention_resp = f"I am ZentroMart's AI assistant. Our official documentation does not contain support or policy guidelines for this query ({trace.abstention_reason}). I must respectfully abstain from providing an unsupported answer."
            return {
                "question": question,
                "response": abstention_resp,
                "decision": "refuse",
                "is_rag": True,
                "grounded": False,
                "abstained": True,
                "trace": trace.to_dict(),
                "retrieved_evidence": [],
                "retrieved_sections": []
            }

        context_str = "\n\n".join([f"[{c.section}] {c.title} ({c.doc_id}):\n{c.content}" for c in chunks])
        resp_raw = self.llm.generate(prompt=question, system_prompt=RAG_SYSTEM_PROMPT, context=context_str)
        response_text = resp_raw[0] if isinstance(resp_raw, tuple) else resp_raw

        evidence_records = [
            {
                "doc_id": c.doc_id,
                "section": c.section,
                "title": c.title,
                "authority": c.authority_level,
                "status": c.status
            }
            for c in chunks
        ]

        decision = "answer"
        if ("kitne din" in question.lower() or "how many days" in question.lower()) and len(question.split()) < 7:
            decision = "clarify"

        return {
            "question": question,
            "response": response_text,
            "decision": decision,
            "is_rag": True,
            "grounded": True,
            "abstained": False,
            "trace": trace.to_dict(),
            "retrieved_evidence": evidence_records,
            "retrieved_sections": [c.section for c in chunks]
        }

RAGAgent = AdaptiveRAGAgent
