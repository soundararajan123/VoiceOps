#!/usr/bin/env python3
"""
VoiceOps 2026: ZentroMart Support Agent — Evaluation-Driven Adaptive RAG Agent
Author: Pearl Lakshmi (pearllakhsmi786)
Repository: https://github.com/pearllakhsmi786/VoiceOps-2026-Project

Usage:
  1. Batch Audio Processing (Official Capstone Command):
     python run.py --input-dir <path_to_audio_folder> --output-dir <path_to_output_folder>

  2. Interactive CLI Chat:
     python run.py --mode chat

  3. Practice Benchmark & Evaluation:
     python run.py --mode evaluate
     python run.py --mode benchmark
"""

import os
import sys
import argparse

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from src.knowledge_base import KnowledgeBase
from src.retriever import AdaptiveRetriever
from src.tools_adapter import ToolsAdapter
from src.llm_adapter import LLMAdapter
from src.baseline import BaselineAgent
from src.rag_agent import AdaptiveRAGAgent
from src.conversation_agent import ConversationAgent
from src.voice_adapter import GnaniVoiceAdapter
from src.benchmark_suite import BenchmarkSuite
from evaluation.eval_questions import run_evaluation as eval_q
from evaluation.eval_conversations import run_evaluation as eval_c
from src.cli import run_interactive_cli, run_voice_cli

def main():
    parser = argparse.ArgumentParser(description="VoiceOps 2026: ZentroMart Support Agent (Evaluation-Driven Adaptive RAG)")
    parser.add_argument("--mode", choices=["chat", "voice", "evaluate", "benchmark", "batch"], default="chat", help="Execution mode")
    parser.add_argument("--strategy", choices=["lexical", "dense", "hybrid", "adaptive"], default="adaptive", help="Retrieval strategy")
    parser.add_argument("--voice", action="store_true", default=False, help="Speak aloud")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K chunks to retrieve")
    parser.add_argument("--dataset-dir", default=os.path.join(cur_dir, "Capstone_Project_Dataset_FTTT_2026"), help="Dataset directory")
    parser.add_argument("--output-dir", default=os.path.join(cur_dir, "sample_outputs"), help="Output directory")
    parser.add_argument("--input-dir", help="Directory containing audio files for batch processing")
    parser.add_argument("--query", help="Single query to test adaptive retrieval and print execution trace")

    args = parser.parse_args()

    # If --input-dir is passed without --mode batch, automatically switch to batch mode
    if args.input_dir:
        args.mode = "batch"

    kb = KnowledgeBase(args.dataset_dir)
    retriever = AdaptiveRetriever(kb, default_strategy=args.strategy, default_top_k=args.top_k)
    tools = ToolsAdapter(args.dataset_dir)
    llm = LLMAdapter(provider="auto")
    agent = ConversationAgent(retriever, tools, llm)
    voice_adapter = GnaniVoiceAdapter(agent)

    if args.query:
        print(f"\n[Adaptive Query] '{args.query}' (Strategy: {args.strategy}, Top-K: {args.top_k})")
        rag = AdaptiveRAGAgent(retriever, llm)
        res = rag.answer_question(args.query, strategy=args.strategy, top_k=args.top_k)
        print(f"\nResponse: {res['response']}")
        print(f"\nTrace: {res['trace']}")
        print(f"Evidence: {res['retrieved_evidence']}")
        return

    if args.mode == "benchmark":
        print("\n=== Running Adaptive RAG Grid Benchmark Suite ===")
        suite = BenchmarkSuite(args.dataset_dir, args.output_dir)
        results = suite.run_grid_experiment()
        print(f"Benchmark finished! Output dashboard: {os.path.join(args.output_dir, 'benchmark_dashboard.html')}")
        print(f"Summary JSON: {os.path.join(args.output_dir, 'benchmark_summary.json')}")

    elif args.mode == "evaluate":
        print("=" * 60)
        print("  Evaluating Step 1 Baseline vs Step 2 Adaptive RAG...")
        print("=" * 60)
        q_metrics = eval_q(args.dataset_dir, args.output_dir)
        print("Practice Questions Summary:")
        print(f"  Step 1 Baseline Decision Accuracy: {q_metrics['step1_baseline']['decision_accuracy_pct']:.1f}%")
        print(f"  Step 2 Adaptive RAG Decision Acc:  {q_metrics['step2_rag']['decision_accuracy_pct']:.1f}%")
        print(f"  Step 2 Fact Coverage Rate:         {q_metrics['step2_rag']['fact_coverage_pct']:.1f}%")
        print(f"  Step 2 Retrieval Evidence Hit:     {q_metrics['step2_rag']['evidence_hit_rate_pct']:.1f}%")

        print("\n" + "=" * 60)
        print("  Evaluating Step 3 Multi-Turn Conversations & Tools...")
        print("=" * 60)
        c_metrics = eval_c(args.dataset_dir, args.output_dir)
        print(f"  Conversations:      {c_metrics['total_conversations']} ({c_metrics['total_turns']} turns)")
        print(f"  Decision Accuracy:  {c_metrics['decision_accuracy_pct']:.1f}%")
        print(f"  Tool-Call Accuracy: {c_metrics['tool_call_accuracy_pct']:.1f}%")
        print(f"  Fact Coverage Rate: {c_metrics['fact_coverage_pct']:.1f}%")
        print(f"\nAll evaluation artifacts saved to: {args.output_dir}")

    elif args.mode == "batch":
        if not args.input_dir:
            print("Error: --input-dir required for batch mode.")
            sys.exit(1)
        print(f"\n=== Processing Batch Audio with Gnani.ai from '{args.input_dir}' ===")
        if not voice_adapter.is_live:
            print("[INFO] GNANI_API_KEY environment variable not detected. Executing batch processor with verification fallback.")
        summary = voice_adapter.process_batch(args.input_dir, args.output_dir)
        print(f"Batch completed: {summary['successful']}/{summary['total_files']} files successfully processed.")
        print(f"Artifacts saved in '{args.output_dir}':")
        print(f"  - Spoken audio replies: *_reply.wav")
        print(f"  - Text replies & retrieved sources: *_reply.txt")
        print(f"  - Audit JSON: batch_processing_summary.json")
        print(f"  - Tabular Results CSV: batch_results.csv")
        sys.exit(0 if summary.get("successful", 0) > 0 or summary.get("failed", 0) == 0 else 1)

    elif args.mode == "voice":
        run_voice_cli(agent, voice_adapter=voice_adapter)
    elif args.mode == "chat":
        run_interactive_cli(agent, voice_adapter=voice_adapter if args.voice else None)
        print("\n" + "="*60)
        print("  ZentroMart Support Agent — Person 6 (Adaptive RAG)")
        print(f"  Strategy: {args.strategy} | Top-K: {args.top_k} | LLM: {llm.provider}")
        print("  Type 'exit' to quit, 'reset' to clear session, 'trace' for last query trace.")
        print("="*60 + "\n")

        while True:
            try:
                user_msg = input("\nCustomer > ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit"]:
                    break
                if user_msg.lower() == "reset":
                    agent.reset()
                    print("[Session reset]")
                    continue
                if user_msg.lower() == "trace":
                    print(f"Last Trace: {retriever.last_trace.to_dict() if retriever.last_trace else 'No trace available'}")
                    continue

                res = agent.process_turn(user_msg)
                print(f"\nAssistant > {res['response']}")
                if res.get("trace", {}).get("tool_call"):
                    print(f"  [Tool Executed]: {res['trace']['tool_call']} -> {res['trace']['tool_result']}")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
