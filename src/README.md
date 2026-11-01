# `src/` — Core Agent Architecture & Modules

This directory contains the source code for the **VoiceOps 2026 ZentroMart Support Agent** by Pearl Lakshmi ([GitHub Repository](https://github.com/pearllakhsmi786/VoiceOps-2026-Project)).

## Module Overview

| File | Purpose | Corresponding Step |
| :--- | :--- | :---: |
| [`knowledge_base.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/knowledge_base.py) | Parses 21 Markdown policy files, preserves `[SECTION-ID]` tags, metadata headers, and establishes document precedence hierarchy (`KB-00.1`). | Step 2 |
| [`retriever.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/retriever.py) | Adaptive retrieval engine supporting BM25 lexical search, Dense TF-IDF cosine similarity, Reciprocal Rank Fusion (RRF) hybrid search, and query classification. | Step 2 |
| [`baseline.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/baseline.py) | Un-grounded prompt-only LLM baseline agent for Step 1 comparative benchmark. | Step 1 |
| [`rag_agent.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/rag_agent.py) | Document-grounded RAG agent with citation tracking, confidence calculation, and principled abstention on out-of-domain queries. | Step 2 |
| [`tools_adapter.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/tools_adapter.py) | Implements relational joins across customers, orders, items, products, pincodes, and holidays, along with `lookup_order` and `create_ticket`. | Step 3 |
| [`conversation_agent.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/conversation_agent.py) | Multi-turn conversational agent with session state memory, identity verification gate (`IDV-1` to `IDV-3`), calendar business-day math, ticket escalation, and `respond(heard)` interface. | Step 3 |
| [`voice_adapter.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/voice_adapter.py) | Gnani.ai STT (`GnaniSTTClient`) and TTS (`GnaniTTSClient`) voice wrapper, speech normalizer (`normalize_text_for_speech`), official `voice_turn(audio_path, agent)`, and batch audio processing. | Step 4 |
| [`benchmark_suite.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/benchmark_suite.py) | Parameter grid experiment evaluator across retrieval strategies (`lexical`, `dense`, `hybrid`, `adaptive`) and Top-K values (3, 5, 7). | Step 2 |
| [`llm_adapter.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/llm_adapter.py) | Multi-provider LLM interface supporting OpenAI, Google Gemini, and offline deterministic responses. | Steps 1-3 |
| [`cli.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/src/cli.py) | Terminal-based interactive text chat and live microphone voice interfaces. | Steps 3 & 4 |
