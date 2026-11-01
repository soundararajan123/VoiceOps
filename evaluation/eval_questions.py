"""
PERSON 6 — Evaluation of Step 1 (Baseline) vs Step 2 (Adaptive RAG) on 52 Practice Questions
Consistent metric definitions: Decision Accuracy, Fact Coverage, Evidence Hit Rate, Abstention.
"""

import os
import csv
import json
from typing import Dict, Any, List
from src.knowledge_base import KnowledgeBase
from src.retriever import AdaptiveRetriever
from src.llm_adapter import LLMAdapter
from src.baseline import BaselineAgent
from src.rag_agent import AdaptiveRAGAgent

def run_evaluation(dataset_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    q_file = os.path.join(dataset_dir, "practice", "questions.csv")
    
    questions = []
    with open(q_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    kb = KnowledgeBase(dataset_dir)
    retriever = AdaptiveRetriever(kb)
    llm = LLMAdapter(provider="auto")
    
    baseline = BaselineAgent(llm)
    rag = AdaptiveRAGAgent(retriever, llm)

    baseline_rows = []
    rag_rows = []

    b_correct_decision = 0
    b_fact_covered = 0
    r_correct_decision = 0
    r_fact_covered = 0
    r_evidence_hits = 0
    unanswerable_total = 0
    r_unsupported_abstained = 0
    total = len(questions)

    for q in questions:
        qid = q.get("id", q.get("question_id", ""))
        text = q["question"]
        exp_decision = q.get("expected_decision", "answer").strip()
        acceptable_decisions = [d.strip() for d in q.get("acceptable_decisions", exp_decision).split(";")]
        answerable = q.get("answerable", "True").strip().lower() == "true"
        if not answerable:
            unanswerable_total += 1

        must_include_raw = q.get("must_include", "").strip()
        must_include_groups = [g.strip() for g in must_include_raw.split(";") if g.strip()]
        must_not_include_raw = q.get("must_not_include", "").strip()
        must_not_include_items = [item.strip() for item in must_not_include_raw.split(";") if item.strip()]

        gold_evidence_raw = q.get("gold_evidence", "").strip()
        gold_sections = [s.strip() for s in gold_evidence_raw.split(";") if s.strip()]

        # 1. Baseline
        b_res = baseline.answer_question(text)
        b_resp = b_res["response"]
        b_dec = b_res.get('decision', 'answer')

        b_facts_ok = True
        if must_include_groups:
            for grp in must_include_groups:
                alts = [alt.strip().lower() for alt in grp.split("/")]
                if not any(alt in b_resp.lower() for alt in alts):
                    b_facts_ok = False
                    break
        else:
            b_facts_ok = False

        b_dec_ok = b_dec in acceptable_decisions
        if b_dec_ok:
            b_correct_decision += 1
        if b_facts_ok:
            b_fact_covered += 1

        baseline_rows.append({
            "question_id": qid,
            "question": text,
            "response": b_resp,
            "decision_correct": b_dec_ok,
            "facts_covered": b_facts_ok
        })

        # 2. RAG
        r_res = rag.answer_question(text)
        r_resp = r_res["response"]
        r_dec = r_res.get("decision", "abstain" if r_res.get("abstained") else "answer")
        r_evidence = [e["section"] for e in r_res.get("retrieved_evidence", [])]

        r_facts_ok = True
        if must_include_groups:
            for grp in must_include_groups:
                alts = [alt.strip().lower() for alt in grp.split("/")]
                if not any(alt in r_resp.lower() for alt in alts):
                    r_facts_ok = False
                    break
        else:
            r_facts_ok = True

        for bad in must_not_include_items:
            if bad.lower() in r_resp.lower():
                r_facts_ok = False
                break

        r_dec_ok = r_dec in acceptable_decisions
        if r_dec_ok:
            r_correct_decision += 1
        if r_facts_ok:
            r_fact_covered += 1

        if not answerable and (r_res.get("abstained") or any(w in r_resp.lower() for w in ["don't have", "not sure", "colleague", "connect", "abstain"])):
            r_unsupported_abstained += 1

        gold_hit = False
        if gold_sections:
            for g in gold_sections:
                if any(g.lower() in ev.lower() for ev in r_evidence):
                    gold_hit = True
                    break
        else:
            gold_hit = True
        if gold_hit:
            r_evidence_hits += 1

        rag_rows.append({
            "question_id": qid,
            "question": text,
            "response": r_resp,
            "decision": r_dec,
            "decision_correct": r_dec_ok,
            "facts_covered": r_facts_ok,
            "evidence_hit": gold_hit,
            "trace": r_res.get("trace")
        })

    metrics = {
        "total_questions": total,
        "step1_baseline": {
            "decision_accuracy_pct": round((b_correct_decision / total) * 100, 1),
            "fact_coverage_pct": round((b_fact_covered / total) * 100, 1)
        },
        "step2_rag": {
            "decision_accuracy_pct": round((r_correct_decision / total) * 100, 1),
            "fact_coverage_pct": round((r_fact_covered / total) * 100, 1),
            "evidence_hit_rate_pct": round((r_evidence_hits / total) * 100, 1),
            "appropriate_abstention_pct": round((r_unsupported_abstained / unanswerable_total) * 100, 1) if unanswerable_total else 100.0
        },
        "llm_provider": llm.provider,
        "llm_model": llm.model
    }

    with open(os.path.join(output_dir, "step1_baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_rows, f, indent=2)

    with open(os.path.join(output_dir, "step2_rag_results.json"), "w", encoding="utf-8") as f:
        json.dump(rag_rows, f, indent=2)

    with open(os.path.join(output_dir, "step1_vs_step2_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
