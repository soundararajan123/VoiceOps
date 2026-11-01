"""
PERSON 6 — Multi-Turn Practice Conversations Evaluation (Step 3)
Evaluates conversation state, tool calls, identity verification, ticket escalation, and error categorization across 23 conversations / 96 turns.
"""

import os
import csv
import json
from collections import defaultdict
from typing import Dict, Any, List

from src.knowledge_base import KnowledgeBase
from src.retriever import AdaptiveRetriever
from src.tools_adapter import ToolsAdapter
from src.llm_adapter import LLMAdapter
from src.conversation_agent import ConversationAgent

def run_evaluation(dataset_dir: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    conv_file = os.path.join(dataset_dir, "practice", "conversations.csv")

    with open(conv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    conversations = defaultdict(list)
    for r in rows:
        conversations[r["conversation_id"]].append(r)

    kb = KnowledgeBase(dataset_dir)
    retriever = AdaptiveRetriever(kb)
    tools = ToolsAdapter(dataset_dir)
    llm = LLMAdapter(provider="auto")
    agent = ConversationAgent(retriever, tools, llm)

    total_turns = len(rows)
    correct_decisions = 0
    correct_tool_calls = 0
    fact_covered = 0
    error_categories_counted = defaultdict(int)
    detailed_results = []

    for cid, turns in conversations.items():
        agent.reset_session(cid)
        conv_turns_res = []

        # Sort turns by turn number
        sorted_turns = sorted(turns, key=lambda t: int(t.get("turn", t.get("turn_id", 0))))

        for t in sorted_turns:
            turn_no = int(t.get("turn", t.get("turn_id", 0)))
            msg = t.get("customer_message", t.get("message", ""))
            exp_decision = t.get("expected_decision", "answer").strip()
            exp_tools_raw = t.get("expected_tool_calls", "").strip()
            exp_tools = [tool.strip() for tool in exp_tools_raw.split(";") if tool.strip()]
            must_inc_raw = t.get("must_include", "").strip()
            must_inc_groups = [g.strip() for g in must_inc_raw.split(";") if g.strip()]
            must_not_raw = t.get("must_not_include", "").strip()
            must_not_items = [item.strip() for item in must_not_raw.split(";") if item.strip()]

            res = agent.process_turn(cid, msg, explicit_order_id=t.get("order_id"))
            actual_decision = res["decision"]
            actual_tools = res["tool_calls"]
            resp_text = res["response"]

            # Decision check
            dec_ok = (actual_decision == exp_decision)
            if dec_ok:
                correct_decisions += 1

            # Tool call check
            tools_ok = True
            if exp_tools:
                for et in exp_tools:
                    tool_base = et.split("(")[0]
                    if not any(tool_base in at for at in actual_tools):
                        tools_ok = False
                        break
            if tools_ok:
                correct_tool_calls += 1

            # Facts check
            facts_ok = True
            if must_inc_groups:
                for grp in must_inc_groups:
                    alts = [alt.strip().lower() for alt in grp.split("/")]
                    if not any(alt in resp_text.lower() for alt in alts):
                        facts_ok = False
                        break
            if must_not_items:
                for bad in must_not_items:
                    if bad.lower() in resp_text.lower():
                        facts_ok = False
                        break
            if facts_ok:
                fact_covered += 1

            err_cat = res.get("trace", {}).get("error_category")
            if err_cat:
                error_categories_counted[err_cat] += 1

            conv_turns_res.append({
                "turn": turn_no,
                "customer_message": msg,
                "assistant_response": resp_text,
                "expected_decision": exp_decision,
                "actual_decision": actual_decision,
                "decision_correct": dec_ok,
                "expected_tools": exp_tools,
                "actual_tools": actual_tools,
                "tools_correct": tools_ok,
                "facts_covered": facts_ok,
                "verified": res.get("verified", False),
                "escalated": res.get("escalated", actual_decision == "escalate"),
                "error_category": err_cat
            })

        detailed_results.append({
            "conversation_id": cid,
            "turns": conv_turns_res
        })

    metrics = {
        "total_conversations": len(conversations),
        "total_turns": total_turns,
        "decision_accuracy_pct": round((correct_decisions / total_turns) * 100, 1),
        "tool_call_accuracy_pct": round((correct_tool_calls / total_turns) * 100, 1),
        "fact_coverage_pct": round((fact_covered / total_turns) * 100, 1),
        "error_categories": dict(error_categories_counted)
    }

    with open(os.path.join(output_dir, "conversations_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "conversations": detailed_results}, f, indent=2)

    return metrics
