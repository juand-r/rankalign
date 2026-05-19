"""cpmi_neg_sweep.py — Sweep gated neg-prompt correction scores on humaneval-v2.1.

Four score families (all at λ=1):

  A. CPMI-neg / entropy gate (original):
       S = Σ_i [ log p_cond[i] - 1{H_i ≥ τ} · log p_neg[i] ]

  B. CPMI-neg / surprisal gate:
       S = Σ_i [ log p_cond[i] - 1{-log p_cond[i] ≥ τ} · log p_neg[i] ]

  C. Both-terms gated / entropy:
       S = Σ_{i: H_i ≥ τ} [ log p_cond[i] - log p_neg[i] ]

  D. Both-terms gated / surprisal:
       S = Σ_{i: -log p_cond[i] ≥ τ} [ log p_cond[i] - log p_neg[i] ]

Special cases (A and B): τ=0 → subtract at all tokens (= tc_neg); τ=∞ → raw log P.
Special cases (C and D): τ=0 → same as tc_neg; τ=∞ → score = 0 (no tokens pass gate).

H_i = entropy_cond[i]   (entropy of the conditional distribution at token i)
Surprisal_i = -log p_cond[i]

Outputs:
    - Terminal table of macro-AUROC vs τ for each family × neg variant
    - CSV: cpmi_neg_sweep_auroc.csv

Usage (from workspace root):
    python scripts/cpmi_neg_sweep.py \\
        --pertok notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl \\
        --output-csv notes/log_P_diff_plots/humaneval-v2.1/cpmi_neg_sweep_auroc.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TAU_GRID_ENTROPY = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
# Surprisal = -log p_cond; typical range 0–20+ nats
TAU_GRID_SURPRISAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
LAMBDA = 1.0
NEG_VARIANTS = ["v1", "v2", "v3"]


def auc_roc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Compute AUROC via Mann-Whitney U (exact, O(n²))."""
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    concordant = sum(1 for p in scores_pos for n in scores_neg if p > n)
    tied = sum(0.5 for p in scores_pos for n in scores_neg if p == n)
    return (concordant + tied) / (n_pos * n_neg)


def macro_auroc(task_scores: dict[str, dict]) -> float:
    """Mean per-task AUROC. task_scores[tid] = {'pos': [...], 'neg': [...]}."""
    aucs = []
    for tid, d in task_scores.items():
        a = auc_roc(d["pos"], d["neg"])
        if not math.isnan(a):
            aucs.append(a)
    return sum(aucs) / len(aucs) if aucs else float("nan")


def score_cpmi_neg(
    logp_cond: list[float],
    gate_vals: list[float],
    logp_neg: list[float],
    tau: float,
) -> float:
    """Family A/B: Σ_i [ log p_cond[i] - 1{gate_vals[i] ≥ τ} · log p_neg[i] ]."""
    n = min(len(logp_cond), len(gate_vals), len(logp_neg))
    total = 0.0
    for i in range(n):
        gate = 1.0 if gate_vals[i] >= tau else 0.0
        total += logp_cond[i] - gate * logp_neg[i]
    return total


def score_both_gated(
    logp_cond: list[float],
    gate_vals: list[float],
    logp_neg: list[float],
    tau: float,
) -> float:
    """Family C/D: Σ_{i: gate_vals[i] ≥ τ} [ log p_cond[i] - log p_neg[i] ]."""
    n = min(len(logp_cond), len(gate_vals), len(logp_neg))
    total = 0.0
    for i in range(n):
        if gate_vals[i] >= tau:
            total += logp_cond[i] - logp_neg[i]
    return total


def load_rows(pertok_path: str) -> list[dict]:
    rows = []
    with open(pertok_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_task_scores(
    rows: list[dict],
    variant: str,
    tau: float,
    score_fn,
    gate_key: str,
) -> dict[str, dict]:
    """Return {task_id: {'pos': [...], 'neg': [...]}} for one config."""
    key_neg = f"logp_neg_{variant}"
    task_scores: dict[str, dict] = defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        if key_neg not in row:
            continue
        if gate_key != "_surprisal" and gate_key not in row:
            continue
        if row["num_tokens"] == 0:
            continue
        if gate_key == "_surprisal":
            gate_vals = [-x for x in row["logp_cond"]]
        else:
            gate_vals = row[gate_key]
        s = score_fn(row["logp_cond"], gate_vals, row[key_neg], tau)
        bucket = "pos" if row["correct"] == "Yes" else "neg"
        task_scores[row["task_id"]][bucket].append(s)
    return dict(task_scores)


def baseline_task_scores(rows: list[dict], lenorm: bool) -> dict:
    """Raw or lenorm baseline (no neg correction)."""
    task_scores: dict[str, dict] = defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        n = row["num_tokens"]
        if n == 0:
            continue
        s = sum(row["logp_cond"])
        if lenorm:
            s /= n
        bucket = "pos" if row["correct"] == "Yes" else "neg"
        task_scores[row["task_id"]][bucket].append(s)
    return dict(task_scores)


def sweep_family(
    rows: list[dict],
    score_fn,
    gate_key: str,
    tau_grid: list[float],
    label: str,
) -> list[dict]:
    """Sweep τ for one scoring family. Returns list of result dicts."""
    results = []
    print(f"\n{'=' * 70}")
    print(f"Family {label}  gate={gate_key}")
    header = f"{'τ':>7}"
    for v in NEG_VARIANTS:
        header += f"  neg_{v:>2}"
    print(header)
    print("-" * (7 + len(NEG_VARIANTS) * 9))

    for tau in tau_grid:
        row_out = {"family": label, "gate": gate_key, "tau": tau}
        line = f"{tau:>7.2f}"
        for v in NEG_VARIANTS:
            auc = macro_auroc(build_task_scores(rows, v, tau, score_fn, gate_key))
            row_out[f"neg_{v}"] = auc
            line += f"  {auc:.4f}"
        print(line)
        results.append(row_out)

    for v in NEG_VARIANTS:
        valid = [r for r in results if not math.isnan(r[f"neg_{v}"])]
        if valid:
            best = max(valid, key=lambda r: r[f"neg_{v}"])
            print(f"  best neg_{v}: {best[f'neg_{v}']:.4f} at τ={best['tau']:.2f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pertok",
        default="notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl",
    )
    ap.add_argument(
        "--output-csv",
        default="notes/log_P_diff_plots/humaneval-v2.1/cpmi_neg_sweep_auroc.csv",
    )
    args = ap.parse_args()

    print(f"Loading {args.pertok} ...")
    rows = load_rows(args.pertok)
    print(f"  {len(rows)} rows loaded.")

    # Baselines
    raw_auc = macro_auroc(baseline_task_scores(rows, lenorm=False))
    lenorm_auc = macro_auroc(baseline_task_scores(rows, lenorm=True))
    tc_neg = {
        v: macro_auroc(
            build_task_scores(
                rows, v, tau=0.0, score_fn=score_cpmi_neg, gate_key="entropy_cond"
            )
        )
        for v in NEG_VARIANTS
    }

    print("\n--- Baselines ---")
    print(f"  raw    : {raw_auc:.4f}")
    print(f"  lenorm : {lenorm_auc:.4f}")
    for v in NEG_VARIANTS:
        print(f"  tc_neg_{v} (τ=0): {tc_neg[v]:.4f}")

    all_results = []

    # A: CPMI-neg / entropy gate
    all_results += sweep_family(
        rows, score_cpmi_neg, "entropy_cond", TAU_GRID_ENTROPY, "A-entropy"
    )

    # B: CPMI-neg / surprisal gate
    all_results += sweep_family(
        rows, score_cpmi_neg, "_surprisal", TAU_GRID_SURPRISAL, "B-surprisal"
    )

    # C: both-terms gated / entropy
    all_results += sweep_family(
        rows, score_both_gated, "entropy_cond", TAU_GRID_ENTROPY, "C-both-entropy"
    )

    # D: both-terms gated / surprisal
    all_results += sweep_family(
        rows, score_both_gated, "_surprisal", TAU_GRID_SURPRISAL, "D-both-surprisal"
    )

    # Summary: best per family × variant
    print(f"\n{'=' * 70}")
    print("SUMMARY — best AUROC per family × neg variant")
    print(f"  raw={raw_auc:.4f}  lenorm={lenorm_auc:.4f}")
    print(f"{'family':<20} {'variant':<8} {'best AUROC':>10}  {'τ':>6}")
    print("-" * 50)
    for label in ["A-entropy", "B-surprisal", "C-both-entropy", "D-both-surprisal"]:
        family_rows = [r for r in all_results if r["family"] == label]
        for v in NEG_VARIANTS:
            valid = [
                r
                for r in family_rows
                if not math.isnan(r.get(f"neg_{v}", float("nan")))
            ]
            if not valid:
                continue
            best = max(valid, key=lambda r: r[f"neg_{v}"])
            print(
                f"  {label:<18} neg_{v}    {best[f'neg_{v}']:>10.4f}  {best['tau']:>6.2f}"
            )

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["family", "gate", "tau"] + [f"neg_{v}" for v in NEG_VARIANTS]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
