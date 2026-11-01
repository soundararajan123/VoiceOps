# VoiceOps 2026: ZentroMart Support Agent
## Architecture: Evaluation-Driven Adaptive RAG Support Agent

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/soundararajan123/VoiceOps)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![IIT Guwahati x Gnani.ai](https://img.shields.io/badge/Capstone-IIT%20Guwahati%20%C3%97%20Gnani.ai-orange.svg)](https://app.gnani.ai/voice)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

- **Public Repository URL:** [https://github.com/soundararajan123/VoiceOps](https://github.com/soundararajan123/VoiceOps)
- **Author:** Soundararajan (`Soundararajan`)
- **Course:** LLM & Generative AI Course · Consulting & Analytics Club, IIT Guwahati × Gnani.ai
- **Capstone Project:** VoiceOps 2026

---

## Official Run Command (Batch Audio Testing)

Per Page 14 of the Capstone Brief (*"One command that takes a folder of customer audio files, runs your agent on each, and saves its replies (as text and audio) with the document chunks it used"*):

```bash
python run.py --input-dir <path_to_audio_folder> --output-dir <path_to_output_folder>
```

### Example:
```bash
python run.py --input-dir sample_outputs/test_audio --output-dir sample_outputs/batch_audio_results
```

For every customer audio file (`<name>.wav`), this command automatically generates:
1. **Synthesized Spoken Reply:** `<name>_reply.wav` (Generated with Gnani TTS using voice *Kaveri*, model *timbre-v2.5*, language *en-IN*)
2. **Text Reply & Retrieved Sources:** `<name>_reply.txt` (Contains customer transcript, agent response, and full retrieved section IDs and chunk excerpts)
3. **Structured JSON Audit:** `batch_processing_summary.json` (Captures per-file status, transcripts, responses, audio paths, and retrieved evidence)
4. **Tabular Results Spreadsheet:** `batch_results.csv` (Spreadsheet summarizing input, transcript, response text, audio output, text output, and sections)

---

## 1. Project Overview & Architecture

This repository delivers an end-to-end, production-grade AI customer support agent for **ZentroMart**, a fictional Indian online store selling groceries and electronics. The agent handles customer inquiries about late deliveries, cancellations, returns, refunds, warranties, and damaged goods via both text and voice.

### Key Architectural Pillars:
1. **Evaluation-Driven Adaptive Retrieval:** Dynamically analyzes customer queries (exact order numbers, SKUs, policy codes vs conversational inquiries) and routes to Okapi BM25 lexical search, Dense TF-IDF cosine similarity, or Reciprocal Rank Fusion (RRF) hybrid search.
2. **Section ID Tracking & Precedence Enforcement (`[KB-00.1]`):** Explicitly preserves document section IDs (e.g., `[RET-2]`, `[WAR-2]`, `[IDV-2]`, `[ESC-1]`) as evidence metadata. Current policy documents outrank agent guides, help articles, and FAQs. Superseded policies (`pol_returns_v3_superseded.md`) are penalized unless queries specifically match the grandfathering clause (electronics delivered before 1 August 2026).
3. **Confidence-Guided Abstention:** Combines dense cosine similarity and lexical token frequency. If confidence falls below threshold (`0.18`), the agent safely abstains with polite clarity rather than hallucinating unsupported claims.
4. **Identity Verification Gate (`[IDV-1]` to `[IDV-3]`):** Strictly protects customer privacy by requiring caller verification (last 4 digits of phone number or 6-digit delivery pincode) before sharing order records, with a maximum of 1 retry.
5. **Deterministic Business Day Math:** Anchors all calculations to the official frozen date **20 October 2026**, counting Monday–Saturday and skipping Sundays and listed warehouse holidays (`holidays.csv`). Return windows strictly observe calendar days.
6. **Voice Integration with Gnani.ai:** High-fidelity speech pipeline wrapping the conversational core with `GnaniSTTClient` (transcription) and `GnaniTTSClient` (synthesis) using official voice **Kaveri** and model **timbre-v2.5**. Includes speech text normalization ("write for the ear").

---

## 2. Directory Structure

```
├── Capstone_Project_Dataset_FTTT_2026/  # Verified dataset (9 CSVs, 21 KB markdown docs, 2 practice CSVs)
│   ├── data/                           # customers, orders, order_items, products, pincodes, holidays, support_logs
│   ├── kb/                             # policy docs, agent guides, help, FAQs, marketing
│   ├── practice/                       # 52 questions.csv, 23 conversations.csv (96 customer turns)
│   ├── tools/                          # lookup_order, create_ticket
│   ├── DATA_DICTIONARY.md              # Column schemas and data dictionary
│   └── validate_dataset.py             # Official dataset validation script
├── src/                                # Core Agent Implementation
│   ├── knowledge_base.py               # Markdown chunker, section ID parser & hierarchy ranker
│   ├── retriever.py                    # BM25 lexical, Dense TF-IDF, RRF hybrid & adaptive routing
│   ├── baseline.py                     # Step 1: Prompt-only ungrounded baseline agent
│   ├── rag_agent.py                    # Step 2: Document-grounded RAG agent with citation tracking
│   ├── tools_adapter.py                # Database queries, identity verification & ticket creation
│   ├── conversation_agent.py           # Step 3: Stateful multi-turn conversational agent with respond()
│   ├── voice_adapter.py                # Step 4: Gnani STT/TTS integration, speech normalizer & batch runner
│   ├── benchmark_suite.py              # Parameter grid evaluation (strategies × top-k)
│   ├── llm_adapter.py                  # Multi-provider LLM adapter (OpenAI, Gemini, deterministic)
│   └── cli.py                          # Interactive terminal chat & voice CLI interfaces
├── evaluation/                         # Evaluation Scripts
│   ├── eval_questions.py               # Step 1 vs Step 2 benchmark on 52 practice questions
│   └── eval_conversations.py           # Step 3 benchmark on 23 practice conversations (96 turns)
├── tests/                              # Comprehensive Test Suite
│   ├── test_agent.py                   # Automated agent unit & integration test suite
│   ├── test_person6.py                 # Core unit tests (retrieval, verification, escalation, voice)
│   └── test_gnani_live.py              # Live Gnani API integration test
├── sample_outputs/                     # Evaluation & Batch Processing Outputs
│   ├── test_audio/                     # Input WAV audio files for batch testing
│   ├── batch_audio_results/            # Generated replies (*_reply.wav, *_reply.txt, summary JSON, CSV)
│   ├── benchmark_dashboard.html        # Interactive HTML grid benchmark dashboard
│   ├── benchmark_summary.json          # Grid benchmark analytics
│   ├── step1_vs_step2_metrics.json     # Practice questions comparison metrics
│   └── conversations_eval_results.json # Multi-turn conversation evaluation metrics
├── submission/                         # Packaged Deliverables
│   └── raja_voiceops.zip       # Clean submission archive (no secrets, no venv, no dataset copy)
├── .env.example                        # Clean environment template with NO secrets
├── requirements.txt                    # Project dependencies
├── run.py                              # Unified multi-mode execution entrypoint
├── generate_report.py                  # Script to generate technical report PDF
├── report.pdf                          # Capstone Technical Report (at most 4 pages)
├── demo_instructions.md                # 2-minute live demo script
└── README.md                           # Project documentation
```

---

## 3. Setup and Installation

### 3.1 Clone Repository & Create Environment
```bash
git clone https://github.com/soundararajan123/VoiceOps.git
cd VoiceOps-2026-Project

# Create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` to configure your credentials:
```bash
# LLM Provider Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_or_gemini_key_here
LLM_MODEL=gpt-4o-mini

# Gnani.ai Speech Services (Step 4 Voice)
# Sign up and obtain key from https://app.gnani.ai/voice
GNANI_API_KEY=your_gnani_api_key_here
GNANI_TTS_VOICE=Kaveri
GNANI_TTS_MODEL=timbre-v2.5
GNANI_LANGUAGE_CODE=en-IN
```

---

## 4. Execution Commands

### A. Batch Audio Processing (Official Run Command)
Processes a folder of customer audio files through Gnani STT, the Step 3 conversational agent, and Gnani TTS:
```bash
python run.py --input-dir sample_outputs/test_audio --output-dir sample_outputs/batch_audio_results
```

### B. Interactive Chat CLI
Interact directly with the agent in your terminal:
```bash
python run.py --mode chat
```

### C. Run Full Evaluation Benchmark
Evaluates all 52 practice questions (Step 1 Baseline vs Step 2 Adaptive RAG) and all 23 multi-turn conversations (Step 3):
```bash
python run.py --mode evaluate
```

### D. Parameter Grid Benchmark Suite & HTML Dashboard
Runs the grid experiment across retrieval modes (`lexical`, `dense`, `hybrid`, `adaptive`) and Top-K values (3, 5, 7):
```bash
python run.py --mode benchmark
```
Outputs an interactive visualization dashboard at `sample_outputs/benchmark_dashboard.html`.

### E. Run Automated Test Suite
```bash
python tests/test_agent.py
```

### F. Rebuild Capstone Technical Report PDF
```bash
python generate_report.py
```

---

## 5. Evaluation Results Summary

| Metric | Step 1 Baseline (Prompt-Only) | Step 2 Adaptive RAG | Step 3 Conversational Agent |
| :--- | :---: | :---: | :---: |
| **Decision Accuracy** | 88.5% | **82.7%** | **96.9%** (93 / 96 turns) |
| **Fact Coverage Rate** | 0.0% (Ungrounded) | **51.9%** (Strict Gold Match) | **96.9%** |
| **Gold Evidence Hit Rate** | 0.0% | **53.8%** (Top-K=5) | **Tool Grounded** |
| **Tool-Call Compliance** | N/A | N/A | **100.0%** (96 / 96 turns) |
| **Identity Verification Shield**| N/A | N/A | **100.0%** (No private data leaks) |
| **Average Query Latency** | 1.2 ms | **2.85 ms** (Adaptive) | **4.1 ms** (Total Turn) |

---

## 6. The Four Steps Implementation

### Step 1: Prompt-Only LLM Baseline
- Implemented in `src/baseline.py`.
- Evaluates the LLM persona without grounding documents. While achieving 88.5% generic decision accuracy on simple questions, its fact coverage was 0.0% due to lack of specific ZentroMart policy numbers.

### Step 2: Document Grounding (Adaptive RAG)
- Implemented in `src/knowledge_base.py`, `src/retriever.py`, and `src/rag_agent.py`.
- Indexes 21 markdown documents preserving section identifiers (`[SECTION-ID]`).
- Precedence hierarchy: Policy > Agent Guide > Help > FAQ > Marketing.
- Preserves superseded policy logic: `pol_returns_v4.md` overrides `pol_returns_v3_superseded.md` except for electronics delivered before 1 August 2026.
- Confidence-guided abstention: when similarity falls below 0.18, safely replies that official documentation does not cover the query.

### Step 3: Multi-Turn Conversational Agent & Tools
- Implemented in `src/conversation_agent.py` and `src/tools_adapter.py`.
- Maintains conversation history and entity state across turns.
- Strict identity verification gate: requires matching last 4 digits of phone or 6-digit pincode before disclosing order data; allows maximum 1 retry.
- Business day calendar logic: anchored to 20 October 2026, counting Mon–Sat and skipping official holidays from `holidays.csv`.
- Human escalation: automatically triggers `create_ticket()` when damaged goods exceed ₹10,000 or upon repeat escalation requests.
- Exposes standard `agent.respond(heard)` interface.

### Step 4: Voice Pipeline with Gnani.ai
- Implemented in `src/voice_adapter.py`.
- Implements Gnani's official Python SDK (`gnani-vachana`):
  - `GnaniSTTClient`: transcribes speech in `en-IN`.
  - `GnaniTTSClient`: synthesizes audio using voice `Kaveri` and model `timbre-v2.5`.
- Text normalization ("write for the ear"): groups order numbers (e.g. `8978 2837`), includes year in dates (`20 October 2026`), and strips markdown/section tags.
- Recovers from silence/unclear audio: asks *"Sorry, I didn't catch that. Could you say it again?"*.
- Provides official `voice_turn(audio_path, agent)` function.

---

## 7. Submission ZIP Structure

As required by Page 14 of the Capstone Brief:

```
raja_voiceops.zip
├── README.md
├── .env.example
├── requirements.txt
├── run.py
├── src/
│   ├── knowledge_base.py
│   ├── retriever.py
│   ├── baseline.py
│   ├── rag_agent.py
│   ├── tools_adapter.py
│   ├── conversation_agent.py
│   ├── voice_adapter.py
│   ├── benchmark_suite.py
│   ├── llm_adapter.py
│   └── cli.py
├── tests/
├── evaluation/
├── report.pdf
├── demo_instructions.md
└── sample_outputs/
    ├── test_audio/
    ├── batch_audio_results/
    ├── benchmark_dashboard.html
    └── benchmark_summary.json
```
*Note: Secrets (`.env`), virtual environments (`venv/`), cache files (`__pycache__/`), and raw dataset copies are strictly excluded.*
