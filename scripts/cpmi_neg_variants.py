"""cpmi_neg_variants.py — Three gating variants with neg-distribution-based gates.

All scores use f2 derived from the NEG model distribution, so the gate condition
reflects the neg model's uncertainty/surprisal rather than the cond model's.

Gate types:
  f1 (for cond term):  cond_entropy = H[p_cond]  |  cond_surprisal = -lpos
  f2 (for neg term):   neg_entropy  = H[p_neg]   |  neg_surprisal  = -lneg

Three variants (notation: lpos = log p_cond, lneg = log p_neg):

  V2 (same tokens, neg gate):
      S = Σ_{i: f2(i)≥τ2} [lpos(i) - λ·lneg(i)]

  V3a (f1=all, neg gate on neg term only):
      S = Σ_i lpos(i)  -  λ · Σ_{i: f2(i)≥τ2} lneg(i)
      (= "Family A/B" but gating on neg distribution instead of cond)

  V3b (independent gates, f1 on cond, f2 on neg):
      S = Σ_{i: f1(i)≥τ1} lpos(i)  -  λ · Σ_{i: f2(i)≥τ2} lneg(i)

  V1 (intersection cond, f2-only neg):
      S = Σ_{i: f1(i)≥τ1 AND f2(i)≥τ2} lpos(i)  -  λ · Σ_{i: f2(i)≥τ2} lneg(i)

Special cases:
  τ2=0 (all tokens pass f2 gate):
    V2 = tc_neg,  V3a = tc_neg,  V3b = Σ_{f1≥τ1} lpos - lneg_total
  τ1=0 in V3b and V1:
    V3b → V3a,  V1 → V2

Usage (from workspace root):
    python scripts/cpmi_neg_variants.py \\
        --pertok notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl \\
        --output-csv notes/log_P_diff_plots/humaneval-v2.1/cpmi_neg_variants_auroc.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Tau grids
TAU_NEG_ENT = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
TAU_NEG_SURP = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
TAU_COND_ENT = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
TAU_COND_SURP = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
NEG_VARIANTS = ["v1", "v2", "v3"]

F2_CONFIGS = {
    "neg_entropy": TAU_NEG_ENT,
    "neg_surprisal": TAU_NEG_SURP,
}
F1_CONFIGS = {
    "cond_entropy": TAU_COND_ENT,
    "cond_surprisal": TAU_COND_SURP,
}


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def auc_roc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney AUROC."""
    if not scores_pos or not scores_neg:
        return float("nan")
    p = np.array(scores_pos)
    n = np.array(scores_neg)
    concordant = int(np.sum(p[:, None] > n[None, :]))
    tied = float(np.sum(p[:, None] == n[None, :])) * 0.5
    return (concordant + tied) / (len(p) * len(n))


def macro_auroc(task_scores: dict) -> float:
    aucs = [auc_roc(d["pos"], d["neg"]) for d in task_scores.values()]
    valid = [a for a in aucs if not math.isnan(a)]
    return sum(valid) / len(valid) if valid else float("nan")


def task_scores_from_row_scores(row_scores: list[tuple[str, str, float]]) -> dict:
    """row_scores: list of (task_id, correct, score). Returns task_scores dict."""
    ts: dict = defaultdict(lambda: {"pos": [], "neg": []})
    for task_id, correct, s in row_scores:
        ts[task_id]["pos" if correct == "Yes" else "neg"].append(s)
    return dict(ts)


# ---------------------------------------------------------------------------
# Data loading and precomputation
# ---------------------------------------------------------------------------


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def precompute(
    rows: list[dict],
    variant: str,
    f1_type: str,
    f2_type: str,
) -> list[dict]:
    """Precompute per-row arrays for all (τ1, τ2) combinations.

    Returns list of dicts:
        task_id, correct, total_cond,
        neg_f2[n_tau2]          — Σ_{f2≥τ2} lneg
        cond_f2[n_tau2]         — Σ_{f2≥τ2} lpos   (V2 numerator)
        cond_f1[n_tau1]         — Σ_{f1≥τ1} lpos   (V3b, V1)
        cond_inter[n_tau1, n_tau2] — Σ_{f1≥τ1 AND f2≥τ2} lpos  (V1)
    """
    key_neg = f"logp_neg_{variant}"
    key_neg_ent = f"entropy_neg_{variant}"
    tau1_grid = F1_CONFIGS[f1_type]
    tau2_grid = F2_CONFIGS[f2_type]

    result = []
    for row in rows:
        if key_neg not in row or row["num_tokens"] == 0:
            continue
        n = min(row["num_tokens"], len(row["logp_cond"]), len(row[key_neg]))
        lp_c = np.array(row["logp_cond"][:n])
        lp_n = np.array(row[key_neg][:n])
        ent_c = np.array(row["entropy_cond"][:n])
        ent_n = np.array(row[key_neg_ent][:n])

        f1 = ent_c if f1_type == "cond_entropy" else -lp_c
        f2 = ent_n if f2_type == "neg_entropy" else -lp_n

        total_cond = float(lp_c.sum())

        # Per-τ2
        neg_f2 = [float(lp_n[f2 >= t].sum()) for t in tau2_grid]
        cond_f2 = [float(lp_c[f2 >= t].sum()) for t in tau2_grid]

        # Per-τ1
        cond_f1 = [float(lp_c[f1 >= t].sum()) for t in tau1_grid]

        # Intersection
        cond_inter = np.zeros((len(tau1_grid), len(tau2_grid)))
        for i1, t1 in enumerate(tau1_grid):
            m1 = f1 >= t1
            for i2, t2 in enumerate(tau2_grid):
                cond_inter[i1, i2] = float(lp_c[m1 & (f2 >= t2)].sum())

        result.append(
            {
                "task_id": row["task_id"],
                "correct": row["correct"],
                "total_cond": total_cond,
                "neg_f2": neg_f2,
                "cond_f2": cond_f2,
                "cond_f1": cond_f1,
                "cond_inter": cond_inter,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Score functions (index into precomputed arrays)
# ---------------------------------------------------------------------------


def _gather(precomp, score_fn) -> list[tuple[str, str, float]]:
    return [(r["task_id"], r["correct"], score_fn(r)) for r in precomp]


def run_v2(precomp, tau2_idx: int, lam: float) -> float:
    """V2: Σ_{f2≥τ2} [lpos - λ·lneg]"""
    rs = _gather(
        precomp, lambda r: r["cond_f2"][tau2_idx] - lam * r["neg_f2"][tau2_idx]
    )
    return macro_auroc(task_scores_from_row_scores(rs))


def run_v3a(precomp, tau2_idx: int, lam: float) -> float:
    """V3a: total_cond - λ · Σ_{f2≥τ2} lneg"""
    rs = _gather(precomp, lambda r: r["total_cond"] - lam * r["neg_f2"][tau2_idx])
    return macro_auroc(task_scores_from_row_scores(rs))


def run_v3b(precomp, tau1_idx: int, tau2_idx: int, lam: float) -> float:
    """V3b: Σ_{f1≥τ1} lpos - λ · Σ_{f2≥τ2} lneg"""
    rs = _gather(
        precomp, lambda r: r["cond_f1"][tau1_idx] - lam * r["neg_f2"][tau2_idx]
    )
    return macro_auroc(task_scores_from_row_scores(rs))


def run_v1(precomp, tau1_idx: int, tau2_idx: int, lam: float) -> float:
    """V1: Σ_{f1≥τ1 AND f2≥τ2} lpos - λ · Σ_{f2≥τ2} lneg"""
    rs = _gather(
        precomp,
        lambda r: r["cond_inter"][tau1_idx, tau2_idx] - lam * r["neg_f2"][tau2_idx],
    )
    return macro_auroc(task_scores_from_row_scores(rs))


# ---------------------------------------------------------------------------
# Grid searches
# ---------------------------------------------------------------------------


def grid_2d(
    precomp: list[dict],
    run_fn,
    tau2_grid: list[float],
    label: str,
) -> list[dict]:
    """2D (τ2, λ) grid. Used for V2 and V3a."""
    results = []
    for t2_idx, tau2 in enumerate(tau2_grid):
        for lam in LAMBDA_GRID:
            auc = run_fn(precomp, t2_idx, lam)
            results.append({"label": label, "tau2": tau2, "lam": lam, "auroc": auc})
    best = max(results, key=lambda r: r["auroc"])
    print(
        f"  {label:<40} best={best['auroc']:.4f}  τ2={best['tau2']:.3f}  λ={best['lam']:.2f}"
    )
    return results


def grid_3d(
    precomp: list[dict],
    run_fn,
    tau1_grid: list[float],
    tau2_grid: list[float],
    label: str,
) -> list[dict]:
    """3D (τ1, τ2, λ) grid. Used for V3b and V1."""
    results = []
    for t1_idx, tau1 in enumerate(tau1_grid):
        for t2_idx, tau2 in enumerate(tau2_grid):
            for lam in LAMBDA_GRID:
                auc = run_fn(precomp, t1_idx, t2_idx, lam)
                results.append(
                    {
                        "label": label,
                        "tau1": tau1,
                        "tau2": tau2,
                        "lam": lam,
                        "auroc": auc,
                    }
                )
    best = max(results, key=lambda r: r["auroc"])
    print(
        f"  {label:<40} best={best['auroc']:.4f}  "
        f"τ1={best['tau1']:.3f}  τ2={best['tau2']:.3f}  λ={best['lam']:.2f}"
    )
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
        default="notes/log_P_diff_plots/humaneval-v2.1/cpmi_neg_variants_auroc.csv",
    )
    args = ap.parse_args()

    print(f"Loading {args.pertok} ...")
    rows = load_rows(args.pertok)
    print(f"  {len(rows)} rows loaded.")

    # Baselines
    def _baseline(lnorm: bool) -> float:
        ts: dict = defaultdict(lambda: {"pos": [], "neg": []})
        for r in rows:
            if r["num_tokens"] == 0:
                continue
            s = sum(r["logp_cond"])
            if lnorm:
                s /= r["num_tokens"]
            ts[r["task_id"]]["pos" if r["correct"] == "Yes" else "neg"].append(s)
        return macro_auroc(dict(ts))

    raw_auc = _baseline(False)
    lenorm_auc = _baseline(True)
    print(f"\nBaselines: raw={raw_auc:.4f}  lenorm={lenorm_auc:.4f}")
    print("Previous best (λ=0.75, τ=0, neg-v1, no gating): 0.8090\n")

    all_results: list[dict] = []

    for neg_v in NEG_VARIANTS:
        print(f"\n{'#' * 70}")
        print(f"NEG VARIANT: {neg_v}")

        for f2_type, tau2_grid in F2_CONFIGS.items():
            print(f"\n  --- f2={f2_type} ---")

            # Precompute with a representative f1 (cond_entropy); intersection also computed
            # We pick one f1 for the precompute; we'll loop over f1_types for V3b/V1.

            # V2 and V3a only need f2, so precompute with f1=cond_entropy (f1 arrays ignored)
            pc_f2only = precompute(rows, neg_v, "cond_entropy", f2_type)

            res = grid_2d(
                pc_f2only,
                lambda pc, t2, lam: run_v2(pc, t2, lam),
                tau2_grid,
                f"V2 | neg_v={neg_v} | f2={f2_type}",
            )
            for r in res:
                r["variant"] = "V2"
                r["neg_v"] = neg_v
                r["f1_type"] = "n/a"
                r["f2_type"] = f2_type
            all_results.extend(res)

            res = grid_2d(
                pc_f2only,
                lambda pc, t2, lam: run_v3a(pc, t2, lam),
                tau2_grid,
                f"V3a | neg_v={neg_v} | f2={f2_type}",
            )
            for r in res:
                r["variant"] = "V3a"
                r["neg_v"] = neg_v
                r["f1_type"] = "all"
                r["f2_type"] = f2_type
            all_results.extend(res)

            for f1_type, tau1_grid in F1_CONFIGS.items():
                pc = precompute(rows, neg_v, f1_type, f2_type)

                res = grid_3d(
                    pc,
                    lambda pc2, t1, t2, lam: run_v3b(pc2, t1, t2, lam),
                    tau1_grid,
                    tau2_grid,
                    f"V3b | neg_v={neg_v} | f1={f1_type} | f2={f2_type}",
                )
                for r in res:
                    r["variant"] = "V3b"
                    r["neg_v"] = neg_v
                    r["f1_type"] = f1_type
                    r["f2_type"] = f2_type
                all_results.extend(res)

                res = grid_3d(
                    pc,
                    lambda pc2, t1, t2, lam: run_v1(pc2, t1, t2, lam),
                    tau1_grid,
                    tau2_grid,
                    f"V1  | neg_v={neg_v} | f1={f1_type} | f2={f2_type}",
                )
                for r in res:
                    r["variant"] = "V1"
                    r["neg_v"] = neg_v
                    r["f1_type"] = f1_type
                    r["f2_type"] = f2_type
                all_results.extend(res)

    # Overall summary by variant × f2_type
    print(f"\n{'=' * 70}")
    print(f"OVERALL BEST  raw={raw_auc:.4f}  lenorm={lenorm_auc:.4f}  prev_best=0.8090")
    print(f"{'config':<50} {'AUROC':>6}")
    print("-" * 58)
    seen: set[str] = set()
    for variant in ["V3a", "V2", "V3b", "V1"]:
        for f2_type in F2_CONFIGS:
            for neg_v in NEG_VARIANTS:
                subset = [
                    r
                    for r in all_results
                    if r["variant"] == variant
                    and r["f2_type"] == f2_type
                    and r["neg_v"] == neg_v
                ]
                if not subset:
                    continue
                best = max(subset, key=lambda r: r["auroc"])
                key = f"{variant}|{neg_v}|{f2_type}"
                if key in seen:
                    continue
                seen.add(key)
                tau1_str = f"τ1={best.get('tau1', '-'):.3f}  " if "tau1" in best else ""
                print(
                    f"  {variant} neg_{neg_v} f2={f2_type:<16} "
                    f"{tau1_str}τ2={best['tau2']:.3f}  λ={best['lam']:.2f}  "
                    f"→ {best['auroc']:.4f}"
                )

    # Write CSV
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "neg_v", "f1_type", "f2_type", "tau1", "tau2", "lam", "auroc"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            if "tau1" not in r:
                r["tau1"] = ""
            w.writerow(r)
    print(f"\nCSV written to: {out}")


if __name__ == "__main__":
    main()
