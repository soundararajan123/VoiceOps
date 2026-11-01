# `tests/` — Automated Test Suite

This directory contains automated unit and integration tests for the **VoiceOps 2026 ZentroMart Support Agent** ([GitHub Repository](https://github.com/pearllakhsmi786/VoiceOps-2026-Project)).

## Test Files

| File | Description |
| :--- | :--- |
| [`test_agent.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/tests/test_agent.py) | Main automated test entrypoint covering full agent capabilities: retrieval, multi-turn, verification, escalation, respond method, and voice pipeline. |
| [`test_person6.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/tests/test_person6.py) | Unit tests verifying adaptive routing, confidence-guided abstention, identity verification gate, ticket creation, speech normalization, and mock voice adapter. |
| [`test_gnani_live.py`](file:///d:/vops/voiceops-all-main/voiceops-all-main/MUTHU/tests/test_gnani_live.py) | Live integration test verifying genuine communication with Gnani.ai STT & TTS APIs when `GNANI_API_KEY` is present in the environment. |

## Running Tests
Run the automated test suite with:
```bash
python tests/test_agent.py
```
Or:
```bash
python tests/test_person6.py
```

To run the live Gnani API connectivity test (requires `GNANI_API_KEY`):
```bash
python tests/test_gnani_live.py
```
