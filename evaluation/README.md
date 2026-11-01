# `evaluation/` — Benchmark & Evaluation Suite

This directory contains evaluation drivers used to benchmark and measure the performance of the **VoiceOps 2026 ZentroMart Support Agent** ([GitHub Repository](https://github.com/pearllakhsmi786/VoiceOps-2026-Project)).

## Scripts

### 1. [`eval_questions.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/evaluation/eval_questions.py)
Evaluates the official 52 practice questions (`practice/questions.csv`) comparing:
- **Step 1 Baseline:** Prompt-only persona without document access.
- **Step 2 Adaptive RAG:** Section-aware retrieval with evidence chunk citations.

**Metrics Evaluated:**
- **Decision Accuracy:** Correct business decision (e.g., answer, clarify, escalate, refuse).
- **Fact Coverage Rate:** Accuracy of critical factual numbers (deadlines, fees, amounts).
- **Gold Evidence Hit Rate:** Retrieval accuracy matching the ground truth `gold_evidence` section IDs.
- **Abstention Rate:** Safe abstention on unsupported out-of-domain queries.

### 2. [`eval_conversations.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/evaluation/eval_conversations.py)
Evaluates all 23 multi-turn customer support conversations (96 customer turns) from `practice/conversations.csv`.

**Metrics Evaluated:**
- **Turn-by-turn Decision Accuracy:** Multi-turn conversational understanding across turns.
- **Tool-Call Compliance:** Correct execution of `lookup_order()` and `create_ticket()`.
- **Identity Verification Enforcement:** Ensuring order data remains shielded until caller identity is verified.
- **Calendar & Holiday Compliance:** Correct calculation of business days anchored to 20 October 2026.

## Running Evaluations
```bash
python run.py --mode evaluate
```
Artifacts are automatically saved to `sample_outputs/`.
