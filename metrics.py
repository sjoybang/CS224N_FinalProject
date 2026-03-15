"""
Computes the three process-oriented evaluation metrics.
"""

import json
import math
import argparse
from pathlib import Path

def rank_to_dist(differential: list) -> dict:
    """Convert ranked differential list → reciprocal-rank probability distribution."""
    if not differential:
        return {}
    scores = {diag: 1.0 / (i + 1) for i, (diag, _) in enumerate(differential)}
    Z = sum(scores.values())
    return {d: s / Z for d, s in scores.items()}


def kl_divergence(p: dict, q: dict, epsilon: float = 1e-8) -> float:
    """KL(p || q) over union of keys, with epsilon smoothing."""
    diagnoses = set(p.keys()) | set(q.keys())
    return sum(
        p.get(d, epsilon) * math.log(p.get(d, epsilon) / q.get(d, epsilon))
        for d in diagnoses
    )

def top_k_accuracy(stage_results: dict, ground_truth: str, k: int) -> float:
    """1 if ground_truth appears in top-k of final stage differential, else 0."""
    final = stage_results.get("stage_4", {}).get("differential", [])
    top_k = [d.lower() for d, _ in final[:k]]
    return float(ground_truth.lower() in top_k)


def reintroduction_error(stage_results: dict) -> float:
    """
    RE = |{d ∈ Et | d ∈ Dt+1}| / |Et|  averaged over all stages t with |Et| > 0.
    Lower is better.
    """
    errors = []
    for t in range(1, 4): 
        elim_t = {d.lower() for d, _ in stage_results.get(f"stage_{t}", {}).get("eliminated", [])}
        if not elim_t:
            continue
        diff_next = {d.lower() for d, _ in stage_results.get(f"stage_{t+1}", {}).get("differential", [])}
        reintroduced = elim_t & diff_next
        errors.append(len(reintroduced) / len(elim_t))

    return sum(errors) / len(errors) if errors else 0.0


def cross_step_consistency(stage_results: dict) -> float:
    """
    Proportion of (diagnosis, stage) elimination pairs where the diagnosis
    stays out of all subsequent differentials.
    Higher is better.
    """
    consistent = 0
    total = 0

    for t in range(1, 4):
        elim_t = {d.lower() for d, _ in stage_results.get(f"stage_{t}", {}).get("eliminated", [])}
        for diag in elim_t:
            total += 1
            stays_out = all(
                diag not in {d.lower() for d, _ in stage_results.get(f"stage_{s}", {}).get("differential", [])}
                for s in range(t + 1, 5)
            )
            if stays_out:
                consistent += 1

    return consistent / total if total > 0 else 1.0


def evidence_sensitivity(base_results: dict, cf_results: dict, altered_stage: int) -> float:
    """
    KL divergence between base and CF distributions at and after the altered stage.
    Averaged over stages [altered_stage .. 4].
    Higher = more belief revision in response to counterfactual evidence.
    """
    kl_values = []
    for t in range(altered_stage, 5):
        key = f"stage_{t}"
        base_diff = base_results.get(key, {}).get("differential", [])
        cf_diff = cf_results.get(key, {}).get("differential", [])
        p = rank_to_dist(base_diff)
        q = rank_to_dist(cf_diff)
        if p and q:
            kl_values.append(kl_divergence(p, q))

    return sum(kl_values) / len(kl_values) if kl_values else 0.0

def compute_metrics(results_dir: Path, model: str) -> dict:
    model_dir = results_dir / model
    result_files = sorted(model_dir.glob("case_*.json"))

    if not result_files:
        print(f"  No results found for model '{model}' in {model_dir}")
        return {}

    top1, top5, re_scores, kl_scores, cons_scores = [], [], [], [], []

    for fpath in result_files:
        with open(fpath) as f:
            case = json.load(f)

        gt = case["ground_truth"]
        base = case["base"]

        top1.append(top_k_accuracy(base, gt, k=1))
        top5.append(top_k_accuracy(base, gt, k=5))
        re_scores.append(reintroduction_error(base))
        cons_scores.append(cross_step_consistency(base))

        for cf in case.get("counterfactuals", []):
            altered = cf.get("altered_stage", 2)
            cf_stages = {k: v for k, v in cf.items() if k.startswith("stage_")}
            kl_scores.append(evidence_sensitivity(base, cf_stages, altered))

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "model": model,
        "n_cases": len(result_files),
        "top1_acc": avg(top1),
        "top5_acc": avg(top5),
        "reintro_error": avg(re_scores),
        "kl_divergence": avg(kl_scores),
        "consistency": avg(cons_scores),
    }

def print_table(rows: list[dict]):
    if not rows:
        print("No results to display.")
        return

    headers = ["Model", "N", "Top-1↑", "Top-5↑", "Reintro↓", "KL↑", "Consist↑"]
    col_w = [12, 6, 8, 8, 10, 8, 10]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + header_line)
    print("-" * len(header_line))
    for row in rows:
        vals = [
            row["model"], row["n_cases"],
            row["top1_acc"], row["top5_acc"],
            row["reintro_error"], row["kl_divergence"],
            row["consistency"],
        ]
        print("  ".join(fmt(v).ljust(w) for v, w in zip(vals, col_w)))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--results_dir", default="data/results",
                        help="Directory containing model result subdirs")
    parser.add_argument("--model", default=None,
                        help="Specific model to evaluate (default: all subdirs)")
    args = parser.parse_args()

    results_path = Path(args.results_dir)

    if args.model:
        models = [args.model]
    else:
        models = [d.name for d in sorted(results_path.iterdir()) if d.is_dir()]

    if not models:
        print(f"No model result directories found in {results_path}")
    else:
        rows = []
        for m in models:
            print(f"Computing metrics for: {m}")
            metrics = compute_metrics(results_path, m)
            if metrics:
                rows.append(metrics)
        print_table(rows)
