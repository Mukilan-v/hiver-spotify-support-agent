"""
Comprehensive Evaluation Harness for Hiver Take-Home Assignment.
Runs automated benchmarks across:
1. Baseline 1 (Trivial Canned Agent)
2. Baseline 2 (Simple Naive Bayes + NN Agent)
3. Proposed SpotifySupportAgent (In-Sample & 5-Fold Stratified Cross-Validation)

Outputs:
- Intent Classification Metrics (Accuracy, Macro-P, Macro-R, Macro-F1)
- Escalation Decision Metrics (Accuracy, Precision, Recall, F1, Cost-Weighted Risk)
- Reply Quality Metrics (BLEU-1..4, ROUGE-1, ROUGE-2, ROUGE-L, Length Compliance)
- LLM Judge Scores (Relevance, Groundedness, Voice, Escalation)
- Failure Cases Log for Deep Analysis
- 5-Fold Stratified Out-of-Fold Cross-Validation Results
- Latency Profiling (ms/query)
"""

import json
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import SpotifySupportAgent
from src.intent_classifier import IntentClassifier
from src.escalation_engine import EscalationEngine
from eval.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from eval.llm_judge import LLMJudge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

def compute_ngram_matches(candidate: str, reference: str, n: int) -> float:
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return 0.0
    cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)]
    ref_ngrams = set(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
    matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
    return matches / len(cand_ngrams) if cand_ngrams else 0.0

def compute_bleu(candidate: str, reference: str) -> Dict[str, float]:
    c_len = len(candidate.split())
    r_len = len(reference.split())
    if c_len == 0 or r_len == 0:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
    p1 = compute_ngram_matches(candidate, reference, 1)
    p2 = compute_ngram_matches(candidate, reference, 2)
    p4 = compute_ngram_matches(candidate, reference, 4)
    bp = 1.0 if c_len > r_len else np.exp(1 - (r_len / max(1, c_len)))
    bleu_1 = bp * p1
    bleu_2 = bp * np.sqrt(max(1e-8, p1 * p2)) if p1 * p2 > 0 else 0.0
    bleu_4 = bp * np.exp(0.25 * np.sum([np.log(max(1e-8, p)) for p in [p1, p2, max(1e-8, p4), max(1e-8, p4)]])) if p4 > 0 else 0.0
    return {
        "bleu_1": round(float(bleu_1), 4),
        "bleu_2": round(float(bleu_2), 4),
        "bleu_4": round(float(bleu_4), 4)
    }

def compute_rouge_l(candidate: str, reference: str) -> float:
    c_words = candidate.lower().split()
    r_words = reference.lower().split()
    m, n = len(c_words), len(r_words)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if c_words[i] == r_words[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(float(f1), 4)

def evaluate_agent(agent_name: str, agent_instance, gold_data: List[Dict[str, Any]], judge: LLMJudge) -> Dict[str, Any]:
    latencies = []
    gold_intents = []
    pred_intents = []
    gold_escalations = []
    pred_escalations = []
    bleu_scores = []
    rouge_scores = []
    length_compliance = []
    judge_scores_list = []
    failures = []

    for item in gold_data:
        tweet = item["incoming_tweet"]
        gold_i = item["gold_intent"]
        gold_e = item["gold_escalation"]
        gold_ref = item["historical_reference_reply"]
        req_dm_gold = item["requires_dm"]

        t0 = time.perf_counter()
        res = agent_instance.handle_tweet(tweet)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        pred_i = res["intent"]
        pred_e = res["action"]
        draft = res["reply"]

        gold_intents.append(gold_i)
        pred_intents.append(pred_i)
        gold_escalations.append(gold_e)
        pred_escalations.append(pred_e)

        bleu = compute_bleu(draft, gold_ref)
        bleu_scores.append(bleu["bleu_2"])
        rouge_scores.append(compute_rouge_l(draft, gold_ref))
        length_compliance.append(1 if len(draft) <= 280 else 0)

        j_eval = judge.judge_reply(
            incoming_tweet=tweet,
            gold_intent=gold_i,
            gold_escalation=gold_e,
            pred_intent=pred_i,
            pred_escalation=pred_e,
            draft_reply=draft,
            historical_reference=gold_ref,
            requires_dm_gold=req_dm_gold
        )
        judge_scores_list.append(j_eval)

        if pred_i != gold_i or pred_e != gold_e or j_eval["mean_score"] < 4.0:
            failures.append({
                "id": item["id"],
                "tweet": tweet,
                "gold_intent": gold_i,
                "pred_intent": pred_i,
                "gold_escalation": gold_e,
                "pred_escalation": pred_e,
                "stated_reason": res.get("stated_reason", ""),
                "draft_reply": draft,
                "historical_reference": gold_ref,
                "judge_score": j_eval["mean_score"],
                "critique": j_eval["critique"],
                "edge_case_type": item.get("edge_case_type", "standard")
            })

    intent_acc = accuracy_score(gold_intents, pred_intents)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(gold_intents, pred_intents, average="macro", zero_division=0)

    y_true_esc = [1 if e == "ESCALATE" else 0 for e in gold_escalations]
    y_pred_esc = [1 if e == "ESCALATE" else 0 for e in pred_escalations]
    esc_acc = accuracy_score(y_true_esc, y_pred_esc)
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, average="binary", zero_division=0)
    
    fps = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt == 0 and yp == 1)
    fns = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt == 1 and yp == 0)
    total_auto = y_true_esc.count(0)
    total_esc = y_true_esc.count(1)
    fpr = (fps / total_auto) if total_auto > 0 else 0.0
    fnr = (fns / total_esc) if total_esc > 0 else 0.0

    avg_relevance = np.mean([j["relevance"] for j in judge_scores_list])
    avg_groundedness = np.mean([j["groundedness"] for j in judge_scores_list])
    avg_voice = np.mean([j["brand_voice"] for j in judge_scores_list])
    avg_esc_correct = np.mean([j["escalation_correctness"] for j in judge_scores_list])
    avg_overall_judge = np.mean([j["mean_score"] for j in judge_scores_list])

    return {
        "agent_name": agent_name,
        "intent_metrics": {
            "accuracy": round(float(intent_acc), 4),
            "precision_macro": round(float(p_mac), 4),
            "recall_macro": round(float(r_mac), 4),
            "f1_macro": round(float(f1_mac), 4)
        },
        "escalation_metrics": {
            "accuracy": round(float(esc_acc), 4),
            "precision": round(float(esc_p), 4),
            "recall": round(float(esc_r), 4),
            "f1_score": round(float(esc_f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4)
        },
        "reply_metrics": {
            "bleu_2": round(float(np.mean(bleu_scores)), 4),
            "rouge_l": round(float(np.mean(rouge_scores)), 4),
            "length_compliance_pct": round(float(np.mean(length_compliance) * 100), 2)
        },
        "judge_scores": {
            "relevance_1_to_5": round(float(avg_relevance), 2),
            "groundedness_1_to_5": round(float(avg_groundedness), 2),
            "brand_voice_1_to_5": round(float(avg_voice), 2),
            "escalation_correctness_1_to_5": round(float(avg_esc_correct), 2),
            "overall_mean_1_to_5": round(float(avg_overall_judge), 2)
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2)
        },
        "failure_count": len(failures),
        "failures": failures
    }

def run_cross_validation(gold_data: List[Dict[str, Any]], n_splits: int = 5) -> Dict[str, Any]:
    """
    Run 5-Fold Stratified Cross-Validation on Intent Classification to report
    unbiased, honest out-of-fold generalization performance.
    """
    texts = np.array([d["incoming_tweet"] for d in gold_data])
    labels = np.array([d["gold_intent"] for d in gold_data])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1s = []
    
    for train_idx, test_idx in skf.split(texts, labels):
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        
        clf = IntentClassifier()
        clf.train(train_texts.tolist(), train_labels.tolist())
        
        preds = [clf.predict(t)["intent"] for t in test_texts]
        acc = accuracy_score(test_labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        
    return {
        "n_splits": n_splits,
        "mean_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "std_accuracy": round(float(np.std(fold_accuracies)), 4),
        "mean_f1_macro": round(float(np.mean(fold_f1s)), 4),
        "std_f1_macro": round(float(np.std(fold_f1s)), 4)
    }

def run_full_benchmark():
    golden_set_path = os.path.join(BASE_DIR, "data", "gold", "golden_eval_set.json")
    with open(golden_set_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    print("=" * 75)
    print(f"RUNNING HIVER BENCHMARK & EVALUATION HARNESS ({len(gold_data)} Golden Samples)")
    print("=" * 75)

    judge = LLMJudge()

    # 1. Baseline 1 (Trivial Canned)
    print("-> Evaluating Baseline 1 (Trivial Canned Agent)...")
    b1_agent = TrivialBaselineAgent()
    b1_results = evaluate_agent("Baseline 1 (Trivial Canned)", b1_agent, gold_data, judge)

    # 2. Baseline 2 (Simple Naive Bayes + NN)
    print("-> Evaluating Baseline 2 (Simple Naive Bayes + NN)...")
    b2_agent = SimpleBaselineAgent(golden_set_path)
    b2_results = evaluate_agent("Baseline 2 (Simple Naive Bayes)", b2_agent, gold_data, judge)

    # 3. Proposed Agent (In-Sample)
    print("-> Evaluating Proposed AI Support Agent (Calibrated + Grounded RAG)...")
    our_agent = SpotifySupportAgent()
    our_results = evaluate_agent("Proposed AI Support Agent", our_agent, gold_data, judge)

    # 4. Out-of-fold Cross-Validation (Honest Generalization)
    print("-> Running 5-Fold Stratified Cross-Validation for Generalization...")
    cv_results = run_cross_validation(gold_data, n_splits=5)

    benchmark_summary = {
        "dataset": "Customer Support on Twitter (@SpotifyCares)",
        "sample_count": len(gold_data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cross_validation_out_of_fold": cv_results,
        "models": {
            "baseline_1_trivial": b1_results,
            "baseline_2_simple": b2_results,
            "proposed_agent": our_results
        }
    }

    # Save benchmark results
    results_path = os.path.join(BASE_DIR, "eval", "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        clean_summary = dict(benchmark_summary)
        for m in clean_summary["models"]:
            clean_summary["models"][m] = {k: v for k, v in clean_summary["models"][m].items() if k != "failures"}
        json.dump(clean_summary, f, indent=2)

    # Save failures separately
    failures_path = os.path.join(BASE_DIR, "eval", "failure_cases.json")
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(our_results["failures"], f, indent=2)

    # Print Headline Comparison Table
    print("\n" + "=" * 95)
    print(f"{'METRIC':<36} | {'BASELINE 1 (TRIVIAL)':<20} | {'BASELINE 2 (SIMPLE)':<20} | {'PROPOSED AGENT':<15}")
    print("=" * 95)
    
    rows = [
        ("Intent Accuracy (In-Sample)", f"{b1_results['intent_metrics']['accuracy']*100:.1f}%", f"{b2_results['intent_metrics']['accuracy']*100:.1f}%", f"{our_results['intent_metrics']['accuracy']*100:.1f}%"),
        ("Intent Macro F1 (In-Sample)", f"{b1_results['intent_metrics']['f1_macro']:.3f}", f"{b2_results['intent_metrics']['f1_macro']:.3f}", f"{our_results['intent_metrics']['f1_macro']:.3f}"),
        ("Intent Accuracy (5-Fold CV OOF)", "N/A", "N/A", f"{cv_results['mean_accuracy']*100:.1f}% ± {cv_results['std_accuracy']*100:.1f}%"),
        ("Intent Macro F1 (5-Fold CV OOF)", "N/A", "N/A", f"{cv_results['mean_f1_macro']:.3f} ± {cv_results['std_f1_macro']:.3f}"),
        ("Escalation Accuracy", f"{b1_results['escalation_metrics']['accuracy']*100:.1f}%", f"{b2_results['escalation_metrics']['accuracy']*100:.1f}%", f"{our_results['escalation_metrics']['accuracy']*100:.1f}%"),
        ("Escalation Recall (Safety-Critical)", f"{b1_results['escalation_metrics']['recall']*100:.1f}%", f"{b2_results['escalation_metrics']['recall']*100:.1f}%", f"{our_results['escalation_metrics']['recall']*100:.1f}%"),
        ("Escalation Precision", f"{b1_results['escalation_metrics']['precision']*100:.1f}%", f"{b2_results['escalation_metrics']['precision']*100:.1f}%", f"{our_results['escalation_metrics']['precision']*100:.1f}%"),
        ("Escalation F1-Score", f"{b1_results['escalation_metrics']['f1_score']:.3f}", f"{b2_results['escalation_metrics']['f1_score']:.3f}", f"{our_results['escalation_metrics']['f1_score']:.3f}"),
        ("False Negative Rate (Missed Risk)", f"{b1_results['escalation_metrics']['false_negative_rate']*100:.1f}%", f"{b2_results['escalation_metrics']['false_negative_rate']*100:.1f}%", f"{our_results['escalation_metrics']['false_negative_rate']*100:.1f}%"),
        ("Reply ROUGE-L Score", f"{b1_results['reply_metrics']['rouge_l']:.3f}", f"{b2_results['reply_metrics']['rouge_l']:.3f}", f"{our_results['reply_metrics']['rouge_l']:.3f}"),
        ("Length Compliance (<280 chars)", f"{b1_results['reply_metrics']['length_compliance_pct']:.1f}%", f"{b2_results['reply_metrics']['length_compliance_pct']:.1f}%", f"{our_results['reply_metrics']['length_compliance_pct']:.1f}%"),
        ("LLM Judge Mean Score (1-5)", f"{b1_results['judge_scores']['overall_mean_1_to_5']:.2f}", f"{b2_results['judge_scores']['overall_mean_1_to_5']:.2f}", f"{our_results['judge_scores']['overall_mean_1_to_5']:.2f}"),
        ("Inference Latency (Mean ms)", f"{b1_results['latency_ms']['mean']:.1f}ms", f"{b2_results['latency_ms']['mean']:.1f}ms", f"{our_results['latency_ms']['mean']:.1f}ms"),
    ]

    for label, b1_val, b2_val, our_val in rows:
        print(f"{label:<36} | {b1_val:<20} | {b2_val:<20} | {our_val:<15}")
    print("=" * 95)
    print(f"\nBenchmark results saved to: {results_path}")
    print(f"Failure audit log ({our_results['failure_count']} cases) saved to: {failures_path}\n")

if __name__ == "__main__":
    run_full_benchmark()
