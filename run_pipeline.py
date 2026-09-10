#!/usr/bin/env python3
"""
CLI Runner for Spotify Customer Support AI Agent.

Usage:
  1. Interactive REPL Mode:
     python run_pipeline.py

  2. Single Tweet Inference:
     python run_pipeline.py --tweet "I was charged twice on my credit card this month!"

  3. Batch File Inference:
     python run_pipeline.py --batch data/gold/golden_eval_set.json --limit 10

  4. Run Benchmark Suite:
     python run_pipeline.py --benchmark
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import SpotifySupportAgent

def format_agent_output(res: dict):
    print("\n" + "=" * 65)
    print(f"CUSTOMER TWEET: \"{res['incoming_tweet']}\"")
    print("-" * 65)
    print(f"CLASSIFIED INTENT : {res['intent']} (Confidence: {res['confidence']:.2f}, Margin: {res['margin']:.2f})")
    
    action_color = "[ESCALATE TO HUMAN]" if res["action"] == "ESCALATE" else "[AUTO-HANDLE]"
    print(f"OPERATIONAL ACTION: {action_color}")
    print(f"STATED REASON     : {res['stated_reason']}")
    print(f"REQUIRES DM (PII) : {'YES' if res['requires_dm'] else 'NO'}")
    print("-" * 65)
    print(f"DRAFTED REPLY ({res['reply_length']} chars, <280 chars: {res['under_280_chars']}):")
    print(f"\"{res['reply']}\"")
    print("=" * 65 + "\n")

def interactive_mode(agent: SpotifySupportAgent):
    print("=" * 65)
    print("Spotify Customer Support AI Agent — Interactive Shell")
    print("Type any simulated customer tweet to see the agent's decision.")
    print("Type 'exit' or 'quit' to exit.")
    print("=" * 65)
    
    while True:
        try:
            tweet = input("\nCustomer Tweet > ").strip()
            if not tweet:
                continue
            if tweet.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break
            res = agent.handle_tweet(tweet)
            format_agent_output(res)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Run Spotify Customer Support AI Agent")
    parser.add_argument("--tweet", type=str, help="Single tweet text to process")
    parser.add_argument("--batch", type=str, help="Path to JSON file containing tweet objects")
    parser.add_argument("--limit", type=int, default=10, help="Maximum items to process in batch mode")
    parser.add_argument("--benchmark", action="store_true", help="Run full evaluation benchmark suite")

    args = parser.parse_args()

    if args.benchmark:
        from eval.evaluate import run_full_benchmark
        run_full_benchmark()
        return

    agent = SpotifySupportAgent()

    if args.tweet:
        res = agent.handle_tweet(args.tweet)
        format_agent_output(res)
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"Error: file not found: {args.batch}")
            sys.exit(1)
        with open(args.batch, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data[:args.limit]
        print(f"Processing {len(items)} items from {args.batch}...")
        for item in items:
            text = item.get("incoming_tweet") or item.get("tweet") or str(item)
            res = agent.handle_tweet(text)
            format_agent_output(res)
    else:
        interactive_mode(agent)

if __name__ == "__main__":
    main()
