# `sample_outputs/` — Evaluation & Batch Processing Artifacts

This directory contains evaluation benchmarks, benchmark dashboards, test audio fixtures, and batch audio outputs for the **VoiceOps 2026 ZentroMart Support Agent** ([GitHub Repository](https://github.com/pearllakhsmi786/VoiceOps-2026-Project)).

## Contents

### 1. `test_audio/`
Input customer WAV audio files used for verifying the official batch audio processing command:
- `sample_wallet_query.wav`: Customer inquiring about ZentroMart wallet refund speed.
- `sample_order_status.wav`: Customer asking for order tracking details.
- `tv_damage_escalation.wav`: Customer reporting high-value TV transit damage requesting escalation.
- `festive_sale_query.wav`: Customer asking about festive season clothing return windows.

### 2. `batch_audio_results/`
Generated output files from running:
```bash
python run.py --input-dir sample_outputs/test_audio --output-dir sample_outputs/batch_audio_results
```
Per Capstone Brief Page 14, this includes:
- `*_reply.wav`: Spoken speech response generated via Gnani TTS (voice *Kaveri*, model *timbre-v2.5*).
- `*_reply.txt`: Formatted text response containing transcription, agent answer, and retrieved document chunks.
- `batch_processing_summary.json`: Detailed JSON audit record of all turns and retrieved evidence.
- `batch_results.csv`: Tabular spreadsheet summarizing batch results.

### 3. Grid Benchmark & Evaluation Artifacts
- `benchmark_dashboard.html`: Interactive HTML dashboard visualizing retrieval precision, latency, and hit rates across strategy and Top-K grid parameters.
- `benchmark_summary.json`: Parameter grid raw metrics.
- `step1_vs_step2_metrics.json`: Comparative metrics on 52 practice questions (Step 1 Baseline vs Step 2 Adaptive RAG).
- `conversations_eval_results.json`: Detailed evaluation log of 23 practice conversations (96 customer turns).
