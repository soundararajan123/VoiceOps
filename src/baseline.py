"""
PERSON 6 — Step 1: Prompt-Only LLM Baseline
"""

from typing import Dict, Any
from .llm_adapter import LLMAdapter

BASELINE_SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for ZentroMart, a leading Indian e-commerce platform.
You must always explicitly identify yourself as an AI assistant at the start of conversations.
You do NOT have access to internal document knowledge bases or customer order databases.
Answer customer questions based purely on general e-commerce knowledge.
Maintain a polite, helpful, and professional tone."""

class BaselineAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
    def answer_question(self, question: str) -> Dict[str, Any]:
        prompt = f"Customer Question: {question}\n\nPlease provide your answer as ZentroMart's AI Support Assistant:"
        resp_raw = self.llm.generate(prompt=prompt, system_prompt=BASELINE_SYSTEM_PROMPT, context=None)
        response_text = resp_raw[0] if isinstance(resp_raw, tuple) else resp_raw
        decision = "answer"
        return {
            "question": question,
            "response": response_text,
            "decision": decision,
            "is_rag": False,
            "retrieved_evidence": [],
            "retrieved_sections": [],
            "source": resp_raw[1] if isinstance(resp_raw, tuple) and len(resp_raw) > 1 else "offline_baseline",
            "has_retrieval": False,
            "grounded": False
        }
