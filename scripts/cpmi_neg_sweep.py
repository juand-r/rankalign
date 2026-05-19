"""cpmi_neg_sweep.py — 2D (τ, λ) grid search over gated neg-prompt correction scores.

Four score families:

  A. CPMI-neg / entropy gate:
       S(τ,λ) = Σ_i [ log p_cond[i] - λ · 1{H_i ≥ τ} · log p_neg[i] ]

  B. CPMI-neg / surprisal gate:
       S(τ,λ) = Σ_i [ log p_cond[i] - λ · 1{surprisal_i ≥ τ} · log p_neg[i] ]

  C. Both-terms gated / entropy:
       S(τ,λ) = Σ_{i: H_i ≥ τ} [ log p_cond[i] - λ · log p_neg[i] ]

  D. Both-terms gated / surprisal:
       S(τ,λ) = Σ_{i: surprisal_i ≥ τ} [ log p_cond[i] - λ · log p_neg[i] ]

where H_i = entropy_cond[i],  surprisal_i = -log p_cond[i].

Families A/B: τ=0,λ=1 → tc_neg;  τ=∞ → raw.
Families C/D: τ=0,λ=1 → tc_neg;  τ=∞ → score=0 (no tokens pass gate).

Efficient implementation: precompute per-row (total_cond, correction_AB(τ),
cond_gated(τ), neg_gated(τ)) so each (τ,λ) evaluation is O(1) per row.

Usage (from workspace root):
    python scripts/cpmi_neg_sweep.py \\
        --pertok notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl \\
        --output-csv notes/log_P_diff_plots/humaneval-v2.1/cpmi_neg_grid_auroc.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TAU_GRID_ENTROPY = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TAU_GRID_SURPRISAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
NEG_VARIANTS = ["v1", "v2", "v3"]


# ---------------------------------------------------------------------------
# AUROC (numpy-accelerated)
# ---------------------------------------------------------------------------


def auc_roc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney AUROC using numpy broadcasting."""
    if not scores_pos or not scores_neg:
        return float("nan")
    p = np.array(scores_pos)
    n = np.array(scores_neg)
    concordant = np.sum(p[:, None] > n[None, :])
    tied = np.sum(p[:, None] == n[None, :]) * 0.5
    return float((concordant + tied) / (len(p) * len(n)))


def macro_auroc(task_scores: dict) -> float:
    """Mean per-task AUROC."""
    aucs = [auc_roc(d["pos"], d["neg"]) for d in task_scores.values()]
    valid = [a for a in aucs if not math.isnan(a)]
    return sum(valid) / len(valid) if valid else float("nan")


# ---------------------------------------------------------------------------
# Data loading and precomputation
# ---------------------------------------------------------------------------


def load_rows(pertok_path: str) -> list[dict]:
    rows = []
    with open(pertok_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def precompute(rows: list[dict], variant: str, gate_type: str) -> list[dict]:
    """Precompute per-row quantities needed for all (τ, λ) evaluations.

    Returns list of dicts with:
        task_id, correct, total_cond,
        correction_per_tau: list[(tau, neg_gated_sum)]   — for families A/B
        cond_gated_per_tau: list[(tau, cond_gated_sum)]  — for families C/D
        neg_gated_per_tau:  list[(tau, neg_gated_sum)]   — for families C/D
    """
    key_neg = f"logp_neg_{variant}"
    tau_grid = TAU_GRID_ENTROPY if gate_type == "entropy" else TAU_GRID_SURPRISAL

    precomp = []
    for row in rows:
        if key_neg not in row or row["num_tokens"] == 0:
            continue

        lp_cond = np.array(row["logp_cond"])
        lp_neg = np.array(row[key_neg])
        n = min(len(lp_cond), len(lp_neg))
        lp_cond = lp_cond[:n]
        lp_neg = lp_neg[:n]

        if gate_type == "entropy":
            gate_vals = np.array(row["entropy_cond"][:n])
        else:
            gate_vals = -lp_cond  # surprisal

        total_cond = float(lp_cond.sum())

        correction_per_tau = []  # Σ_{gated} log p_neg  (for A/B)
        cond_gated_per_tau = []  # Σ_{gated} log p_cond (for C/D)
        neg_gated_per_tau = []  # Σ_{gated} log p_neg  (for C/D)

        for tau in tau_grid:
            mask = gate_vals >= tau
            correction_per_tau.append(float(lp_neg[mask].sum()))
            cond_gated_per_tau.append(float(lp_cond[mask].sum()))
            neg_gated_per_tau.append(float(lp_neg[mask].sum()))

        precomp.append(
            {
                "task_id": row["task_id"],
                "correct": row["correct"],
                "total_cond": total_cond,
                "correction_per_tau": correction_per_tau,
                "cond_gated_per_tau": cond_gated_per_tau,
                "neg_gated_per_tau": neg_gated_per_tau,
            }
        )
    return precomp


def scores_AB(precomp: list[dict], tau_idx: int, lam: float) -> dict:
    """S(τ,λ) = total_cond - λ · correction(τ).  Families A and B."""
    task_scores: dict = defaultdict(lambda: {"pos": [], "neg": []})
    for r in precomp:
        s = r["total_cond"] - lam * r["correction_per_tau"][tau_idx]
        bucket = "pos" if r["correct"] == "Yes" else "neg"
        task_scores[r["task_id"]][bucket].append(s)
    return dict(task_scores)


def scores_CD(precomp: list[dict], tau_idx: int, lam: float) -> dict:
    """S(τ,λ) = cond_gated(τ) - λ · neg_gated(τ).  Families C and D."""
    task_scores: dict = defaultdict(lambda: {"pos": [], "neg": []})
    for r in precomp:
        s = r["cond_gated_per_tau"][tau_idx] - lam * r["neg_gated_per_tau"][tau_idx]
        bucket = "pos" if r["correct"] == "Yes" else "neg"
        task_scores[r["task_id"]][bucket].append(s)
    return dict(task_scores)


# ---------------------------------------------------------------------------
# Grid search for one family
# ---------------------------------------------------------------------------


def grid_search(
    precomp: list[dict],
    score_fn,
    tau_grid: list[float],
    label: str,
) -> list[dict]:
    """Run 2D (τ, λ) grid search. Returns list of result dicts."""
    results = []
    print(f"\n{'=' * 70}")
    print(f"Family {label}")
    # Header
    lam_cols = "  ".join(f"λ={lam:<4}" for lam in LAMBDA_GRID)
    print(f"{'τ':>7}  {lam_cols}")
    print("-" * (7 + len(LAMBDA_GRID) * 10))

    for tau_idx, tau in enumerate(tau_grid):
        line = f"{tau:>7.2f}"
        for lam in LAMBDA_GRID:
            auc = macro_auroc(score_fn(precomp, tau_idx, lam))
            results.append({"family": label, "tau": tau, "lambda": lam, "auroc": auc})
            line += f"  {auc:.4f}"
        print(line)

    best = max(results, key=lambda r: r["auroc"])
    print(f"  → best: {best['auroc']:.4f}  τ={best['tau']:.2f}  λ={best['lambda']:.2f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pertok",
        default="notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl",
    )
    ap.add_argument(
        "--output-csv",
        default="notes/log_P_diff_plots/humaneval-v2.1/cpmi_neg_grid_auroc.csv",
    )
    args = ap.parse_args()

    print(f"Loading {args.pertok} ...")
    rows = load_rows(args.pertok)
    print(f"  {len(rows)} rows loaded.")

    def _baseline(lnorm: bool) -> dict:
        ts: dict = defaultdict(lambda: {"pos": [], "neg": []})
        for r in rows:
            if r["num_tokens"] == 0:
                continue
            s = sum(r["logp_cond"])
            if lnorm:
                s /= r["num_tokens"]
            ts[r["task_id"]]["pos" if r["correct"] == "Yes" else "neg"].append(s)
        return dict(ts)

    raw_auc = macro_auroc(_baseline(False))
    lenorm_auc = macro_auroc(_baseline(True))

    tc_neg = {}
    for v in NEG_VARIANTS:
        pc = precompute(rows, v, "entropy")
        tc_neg[v] = macro_auroc(scores_AB(pc, tau_idx=0, lam=1.0))

    print("\n--- Baselines ---")
    print(f"  raw    : {raw_auc:.4f}")
    print(f"  lenorm : {lenorm_auc:.4f}")
    for v in NEG_VARIANTS:
        print(f"  tc_neg_{v}: {tc_neg[v]:.4f}")

    all_results = []

    for v in NEG_VARIANTS:
        print(f"\n{'#' * 70}")
        print(f"NEG VARIANT: {v}")

        # Precompute once per (variant, gate_type)
        pc_ent = precompute(rows, v, "entropy")
        pc_surp = precompute(rows, v, "surprisal")

        res_A = grid_search(pc_ent, scores_AB, TAU_GRID_ENTROPY, f"A-entropy-{v}")
        res_B = grid_search(pc_surp, scores_AB, TAU_GRID_SURPRISAL, f"B-surprisal-{v}")
        res_C = grid_search(pc_ent, scores_CD, TAU_GRID_ENTROPY, f"C-both-entropy-{v}")
        res_D = grid_search(
            pc_surp, scores_CD, TAU_GRID_SURPRISAL, f"D-both-surprisal-{v}"
        )

        for r in res_A + res_B + res_C + res_D:
            r["variant"] = v
        all_results += res_A + res_B + res_C + res_D

    # Overall summary
    print(f"\n{'=' * 70}")
    print(f"OVERALL BEST (raw={raw_auc:.4f}  lenorm={lenorm_auc:.4f})")
    for fam_prefix in [
        "A-entropy",
        "B-surprisal",
        "C-both-entropy",
        "D-both-surprisal",
    ]:
        for v in NEG_VARIANTS:
            label = f"{fam_prefix}-{v}"
            subset = [r for r in all_results if r["family"] == label]
            if not subset:
                continue
            best = max(subset, key=lambda r: r["auroc"])
            print(
                f"  {label:<28} best={best['auroc']:.4f}  "
                f"τ={best['tau']:.2f}  λ={best['lambda']:.2f}"
            )

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["family", "variant", "tau", "lambda", "auroc"]
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
