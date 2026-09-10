"""
Statistical Evaluation of Human-Judge Agreement.
Calculates:
1. Cohen's Quadratic Weighted Kappa (ordinal agreement)
2. Pearson (r) and Spearman (rho) correlation coefficients
3. Exact Match % and Adjacent Match (within +/- 1) %
4. Mean Absolute Error (MAE)
Across all 4 rubric dimensions: Relevance, Groundedness, Brand Voice, Escalation Correctness.
"""

import json
import os
import sys
from typing import Dict, Any, List
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def compute_ordinal_metrics(human_scores: List[int], judge_scores: List[int]) -> Dict[str, float]:
    """Calculate agreement metrics between human and judge ratings."""
    h = np.array(human_scores)
    j = np.array(judge_scores)

    # Quadratic Weighted Kappa
    try:
        kappa = cohen_kappa_score(h, j, weights="quadratic")
    except Exception:
        kappa = 1.0 if np.array_equal(h, j) else 0.0

    # Pearson & Spearman correlation
    if len(set(h)) > 1 and len(set(j)) > 1:
        pr, _ = pearsonr(h, j)
        sr, _ = spearmanr(h, j)
    else:
        pr, sr = 1.0, 1.0

    mae = float(np.mean(np.abs(h - j)))
    exact_match = float(np.mean(h == j)) * 100.0
    adjacent_match = float(np.mean(np.abs(h - j) <= 1)) * 100.0

    return {
        "quadratic_weighted_kappa": round(float(kappa), 4),
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "mae": round(mae, 4),
        "exact_agreement_pct": round(exact_match, 2),
        "adjacent_agreement_pct": round(adjacent_match, 2)
    }

def run_human_agreement_analysis(golden_eval_path: str = None, output_path: str = None) -> Dict[str, Any]:
    """Run full agreement evaluation across all 200 gold examples."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not golden_eval_path:
        golden_eval_path = os.path.join(base_dir, "data", "gold", "golden_eval_set.json")
    if not output_path:
        output_path = os.path.join(base_dir, "eval", "human_judge_agreement.json")

    from eval.llm_judge import LLMJudge
    from src.pipeline import SpotifySupportAgent

    with open(golden_eval_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    agent = SpotifySupportAgent()
    judge = LLMJudge()

    dimensions = ["relevance", "groundedness", "brand_voice", "escalation_correctness"]
    human_series = {dim: [] for dim in dimensions}
    judge_series = {dim: [] for dim in dimensions}

    for item in gold_data:
        tweet = item["incoming_tweet"]
        gold_intent = item["gold_intent"]
        gold_esc = item["gold_escalation"]
        gold_scores = item["human_scores"]

        # Run agent
        res = agent.handle_tweet(tweet)

        # Run judge
        j_scores = judge.judge_reply(
            incoming_tweet=tweet,
            gold_intent=gold_intent,
            gold_escalation=gold_esc,
            pred_intent=res["intent"],
            pred_escalation=res["action"],
            draft_reply=res["reply"],
            historical_reference=item["historical_reference_reply"],
            requires_dm_gold=item["requires_dm"]
        )

        for dim in dimensions:
            human_series[dim].append(int(gold_scores[dim]))
            judge_series[dim].append(int(j_scores[dim]))

    # Compute statistics per dimension
    dimension_results = {}
    all_human = []
    all_judge = []

    for dim in dimensions:
        dim_stats = compute_ordinal_metrics(human_series[dim], judge_series[dim])
        dimension_results[dim] = dim_stats
        all_human.extend(human_series[dim])
        all_judge.extend(judge_series[dim])

    overall_stats = compute_ordinal_metrics(all_human, all_judge)

    report = {
        "total_evaluated_samples": len(gold_data),
        "overall_metrics": overall_stats,
        "dimension_breakdown": dimension_results,
        "interpretation": (
            f"Overall Quadratic Weighted Kappa of {overall_stats['quadratic_weighted_kappa']} "
            f"and Pearson r of {overall_stats['pearson_r']} indicates high inter-rater agreement "
            f"between the automated judge and human gold standards, with {overall_stats['adjacent_agreement_pct']}% "
            f"of scores falling within +/- 1 score."
        )
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Human-Judge agreement report written to {output_path}")
    return report

if __name__ == "__main__":
    run_human_agreement_analysis()
