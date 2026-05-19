"""cpmi_neg_sweep.py — Sweep CPMI-neg score on humaneval-v2.1.

Formula:
    S(τ, λ) = Σ_i [ log p_cond[i] - λ · 1{H_i ≥ τ} · log p_neg[i] ]
    where H_i = entropy_cond[i]

Special cases:
    τ = 0  → subtracts λ·log p_neg at every token  (= λ · tc_neg when λ=1)
    τ = ∞  → no subtraction at all                  (= raw log P)

Length-normalized variant: S(τ, λ) / N

Outputs:
    - Terminal table of macro-AUROC vs τ for each neg variant
    - CSV: cpmi_neg_sweep_auroc.csv
    - Comparison row against baselines (raw, lenorm, tc_neg_v1)

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


TAU_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
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


def cpmi_neg_score(
    logp_cond: list[float],
    entropy_cond: list[float],
    logp_neg: list[float],
    tau: float,
    lam: float = 1.0,
) -> float:
    """Compute CPMI-neg score for one completion."""
    n = min(len(logp_cond), len(entropy_cond), len(logp_neg))
    total = 0.0
    for i in range(n):
        gate = 1.0 if entropy_cond[i] >= tau else 0.0
        total += logp_cond[i] - lam * gate * logp_neg[i]
    return total


def load_rows(pertok_path: str) -> list[dict]:
    rows = []
    with open(pertok_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_task_scores_for_variant(
    rows: list[dict], variant: str, tau: float, lam: float, lenorm: bool
) -> dict[str, dict]:
    """Return {task_id: {'pos': [...], 'neg': [...]}} for one (variant, τ) combo."""
    key_neg = f"logp_neg_{variant}"
    task_scores: dict[str, dict] = defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        if key_neg not in row:
            continue
        n = row["num_tokens"]
        if n == 0:
            continue
        s = cpmi_neg_score(
            row["logp_cond"],
            row["entropy_cond"],
            row[key_neg],
            tau=tau,
            lam=lam,
        )
        if lenorm:
            s /= n
        bucket = "pos" if row["correct"] == "Yes" else "neg"
        task_scores[row["task_id"]][bucket].append(s)
    return dict(task_scores)


def baseline_task_scores(rows: list[dict], variant: str, lenorm: bool) -> dict:
    """raw or lenorm baseline using only logp_cond."""
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
    ap.add_argument("--lam", type=float, default=LAMBDA)
    args = ap.parse_args()

    print(f"Loading {args.pertok} ...")
    rows = load_rows(args.pertok)
    print(f"  {len(rows)} rows loaded.")

    # Baselines
    raw_auc = macro_auroc(baseline_task_scores(rows, "raw", lenorm=False))
    lenorm_auc = macro_auroc(baseline_task_scores(rows, "lenorm", lenorm=True))

    # tc_neg baselines (τ=0, all tokens subtracted)
    tc_neg = {}
    tc_neg_len = {}
    for v in NEG_VARIANTS:
        tc_neg[v] = macro_auroc(
            build_task_scores_for_variant(rows, v, tau=0.0, lam=args.lam, lenorm=False)
        )
        tc_neg_len[v] = macro_auroc(
            build_task_scores_for_variant(rows, v, tau=0.0, lam=args.lam, lenorm=True)
        )

    print("\nBaselines:")
    print(f"  raw             : {raw_auc:.4f}")
    print(f"  lenorm          : {lenorm_auc:.4f}")
    for v in NEG_VARIANTS:
        print(f"  tc_neg_{v}      : {tc_neg[v]:.4f}  (lenorm: {tc_neg_len[v]:.4f})")

    # CPMI-neg sweep
    results = []
    print(f"\nCPMI-neg sweep (λ={args.lam}):")
    print(f"{'τ':>6}", end="")
    for v in NEG_VARIANTS:
        print(f"  {'neg_' + v:>10}  {'neg_' + v + '_len':>14}", end="")
    print()
    print("-" * (6 + len(NEG_VARIANTS) * 28))

    for tau in TAU_GRID:
        row_out = {"tau": tau}
        print(f"{tau:>6.2f}", end="")
        for v in NEG_VARIANTS:
            auc = macro_auroc(
                build_task_scores_for_variant(
                    rows, v, tau=tau, lam=args.lam, lenorm=False
                )
            )
            auc_len = macro_auroc(
                build_task_scores_for_variant(
                    rows, v, tau=tau, lam=args.lam, lenorm=True
                )
            )
            row_out[f"neg_{v}"] = auc
            row_out[f"neg_{v}_len"] = auc_len
            print(f"  {auc:>10.4f}  {auc_len:>14.4f}", end="")
        print()
        results.append(row_out)

    # Best values
    print("\nBest across τ grid:")
    for v in NEG_VARIANTS:
        best = max(results, key=lambda r: r[f"neg_{v}"])
        best_len = max(results, key=lambda r: r[f"neg_{v}_len"])
        print(
            f"  neg_{v}      best={best[f'neg_{v}']:.4f} at τ={best['tau']:.2f}  "
            f"lenorm best={best_len[f'neg_{v}_len']:.4f} at τ={best_len['tau']:.2f}"
        )

    print(f"\n  lenorm baseline: {lenorm_auc:.4f}  raw baseline: {raw_auc:.4f}")

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tau"] + [
        f"neg_{v}{suf}" for v in NEG_VARIANTS for suf in ["", "_len"]
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written to: {out_path}")

    # Also append baseline rows for easy reference
    baselines_csv = out_path.with_name(out_path.stem + "_baselines.csv")
    with open(baselines_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "auroc"])
        writer.writeheader()
        writer.writerow({"metric": "raw", "auroc": raw_auc})
        writer.writerow({"metric": "lenorm", "auroc": lenorm_auc})
        for v in NEG_VARIANTS:
            writer.writerow({"metric": f"tc_neg_{v}", "auroc": tc_neg[v]})
            writer.writerow({"metric": f"tc_neg_{v}_len", "auroc": tc_neg_len[v]})
    print(f"Baselines written to: {baselines_csv}")


if __name__ == "__main__":
    main()
