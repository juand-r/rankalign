"""analyze_wrong_examples.py — per-token error analysis on wrong humaneval-v2.1 examples.

For each selected task:
  1. Join per-token logp arrays (from humaneval_v2_1_pertok_scores.jsonl) with answer text
     (from v2.1 CSVs) via task_id + row_idx.
  2. Tokenize answers using the gemma-4-31B-it tokenizer (same tokenizer used during scoring).
  3. For each wrong example: align tokens with logp_cond and logp_neg_v1, identify
     low-logp (surprise) positions, and check if logp_neg_v1[i] > logp_cond[i] there.
  4. Print a terminal report and write a markdown file.

Usage:
    python scripts/analyze_wrong_examples.py \\
        --pertok notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl \\
        --v21-dir data/humaneval/v2.1 \\
        --tasks humaneval_130 humaneval_122 \\
        --max-wrong 3 \\
        --output-md notes/log_P_diff_plots/humaneval-v2.1/wrong_examples_analysis.md

Alignment contract:
  scoring used:  completion_ids = tok(answer, add_special_tokens=False)["input_ids"]
  This script:   same tokenizer call → same token sequence → same indices as logp arrays.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def load_pertok_for_tasks(pertok_path: str, task_ids: list[str]) -> dict:
    """Return {task_id: {row_idx: row_dict}} for requested tasks."""
    wanted = set(task_ids)
    result: dict[str, dict[int, dict]] = {t: {} for t in wanted}
    with open(pertok_path) as f:
        for line in f:
            row = json.loads(line)
            tid = row["task_id"]
            if tid in wanted:
                result[tid][row["row_idx"]] = row
    return result


def load_csv_rows(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_tokenizer(model_id: str = "google/gemma-4-31B-it") -> object:
    """Load tokenizer from HF cache (tokenizer-only, no model weights needed)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def tokenize_answer(tok, answer: str) -> list[str]:
    """Tokenize answer text (add_special_tokens=False) and return token strings."""
    ids = tok(answer, add_special_tokens=False)["input_ids"]
    return tok.convert_ids_to_tokens(ids)


def summarize_locality(logp: list[float], percentile: float = 10.0) -> dict:
    """Compute locality stats for a logp array.

    Returns:
        frac_low: fraction of tokens below the percentile threshold
        mass_frac: fraction of total surprisal (−logp) concentrated in bottom-percentile tokens
        is_localized: True if the bottom 10% of tokens account for >50% of total surprisal
        sorted_surprisal_frac: list of (rank, frac_of_total) for cumulative Lorenz-style curve
    """
    surp = [-lp for lp in logp]
    total = sum(surp)
    n = len(surp)
    if n == 0 or total == 0:
        return {"frac_low": 0, "mass_frac": 0, "is_localized": False}

    sorted_surp = sorted(surp, reverse=True)
    k = max(1, math.ceil(n * percentile / 100))
    top_k_mass = sum(sorted_surp[:k])
    mass_frac = top_k_mass / total

    return {
        "frac_low_tokens": k / n,
        "mass_frac": mass_frac,
        "is_localized": mass_frac > 0.5,
        "total_surprisal": total,
        "n_tokens": n,
        "mean_logp": sum(logp) / n,
    }


def neg_advantage_at_errors(
    logp_cond: list[float],
    logp_neg: list[float],
    error_threshold_z: float = -1.5,
) -> dict:
    """Check if neg prompt raises probability at low-logp (error) positions.

    A token is an "error token" if its logp_cond is more than error_threshold_z
    standard deviations below the mean of the completion.

    Returns:
        n_error_tokens: count of error tokens
        frac_error_tokens: fraction of tokens that are error tokens
        mean_delta_error: mean (logp_neg - logp_cond) at error positions (positive = neg helps)
        mean_delta_normal: mean (logp_neg - logp_cond) at non-error positions
        neg_helps_at_errors: True if mean_delta_error > mean_delta_normal
    """
    import statistics

    n = len(logp_cond)
    if n < 3:
        return {
            "n_error_tokens": 0,
            "frac_error_tokens": 0,
            "mean_delta_error": 0,
            "mean_delta_normal": 0,
            "neg_helps_at_errors": False,
        }

    mu = statistics.mean(logp_cond)
    sigma = statistics.stdev(logp_cond)
    if sigma == 0:
        return {
            "n_error_tokens": 0,
            "frac_error_tokens": 0,
            "mean_delta_error": 0,
            "mean_delta_normal": 0,
            "neg_helps_at_errors": False,
        }

    error_mask = [(lp - mu) / sigma < error_threshold_z for lp in logp_cond]
    deltas = [logp_neg[i] - logp_cond[i] for i in range(n)]

    err_deltas = [deltas[i] for i in range(n) if error_mask[i]]
    norm_deltas = [deltas[i] for i in range(n) if not error_mask[i]]

    mean_err = sum(err_deltas) / len(err_deltas) if err_deltas else float("nan")
    mean_norm = sum(norm_deltas) / len(norm_deltas) if norm_deltas else float("nan")

    return {
        "n_error_tokens": sum(error_mask),
        "frac_error_tokens": sum(error_mask) / n,
        "mean_delta_error": mean_err,
        "mean_delta_normal": mean_norm,
        "neg_helps_at_errors": (
            not (math.isnan(mean_err) or math.isnan(mean_norm)) and mean_err > mean_norm
        ),
    }


def format_token_table(
    tokens: list[str],
    logp_cond: list[float],
    logp_neg: list[float],
    max_tokens: int = 50,
) -> str:
    """Return a fixed-width table of token | logp_cond | logp_neg | delta."""
    lines = [f"{'TOKEN':<20} {'logp_cond':>10} {'logp_neg_v1':>12} {'delta':>8}  FLAG"]
    lines.append("-" * 60)
    n = min(len(tokens), len(logp_cond), len(logp_neg), max_tokens)
    import statistics

    mu = statistics.mean(logp_cond[:n]) if n > 0 else 0
    sigma = statistics.stdev(logp_cond[:n]) if n > 1 else 1e-9
    for i in range(n):
        tok = repr(tokens[i])[:18]
        lpc = logp_cond[i]
        lpn = logp_neg[i]
        d = lpn - lpc
        z = (lpc - mu) / sigma if sigma > 0 else 0
        flag = " ← ERROR" if z < -1.5 else ""
        if d > 0.5 and z < -1.5:
            flag += " +NEG"
        lines.append(f"{tok:<20} {lpc:>10.3f} {lpn:>12.3f} {d:>8.3f}{flag}")
    if len(tokens) > max_tokens:
        lines.append(f"  ... ({len(tokens) - max_tokens} more tokens not shown)")
    return "\n".join(lines)


def analyze_example(
    tok,
    csv_row: dict,
    pertok_row: dict,
    task_id: str,
    row_idx: int,
    max_table_tokens: int = 60,
) -> dict:
    """Full analysis of one wrong example. Returns analysis dict."""
    answer = csv_row.get("answer", "") or ""
    question = csv_row.get("question", "") or ""
    model = csv_row.get("model", "?")
    strategy = csv_row.get("strategy", "?")

    tokens = tokenize_answer(tok, answer)
    logp_cond = pertok_row["logp_cond"]
    logp_neg_v1 = pertok_row["logp_neg_v1"]

    # Verify alignment
    n_stored = pertok_row["num_tokens"]
    n_tok = len(tokens)
    mismatch = n_tok != n_stored
    if mismatch:
        print(
            f"  WARNING: {task_id}:{row_idx} token count mismatch: "
            f"stored={n_stored}, retokenized={n_tok}",
            file=sys.stderr,
        )

    # Use the shorter length for safety
    n = min(n_tok, len(logp_cond), len(logp_neg_v1))
    tokens = tokens[:n]
    logp_cond = logp_cond[:n]
    logp_neg_v1 = logp_neg_v1[:n]

    locality = summarize_locality(logp_cond)
    neg_adv = neg_advantage_at_errors(logp_cond, logp_neg_v1)
    token_table = format_token_table(tokens, logp_cond, logp_neg_v1, max_table_tokens)

    # Also compute neg_adv for neg_v2, neg_v3 if present
    neg_adv_v2 = (
        neg_advantage_at_errors(logp_cond, pertok_row["logp_neg_v2"][:n])
        if "logp_neg_v2" in pertok_row
        else None
    )
    neg_adv_v3 = (
        neg_advantage_at_errors(logp_cond, pertok_row["logp_neg_v3"][:n])
        if "logp_neg_v3" in pertok_row
        else None
    )

    return {
        "task_id": task_id,
        "row_idx": row_idx,
        "model": model,
        "strategy": strategy,
        "n_tokens": n,
        "token_mismatch": mismatch,
        "mean_logp_cond": sum(logp_cond) / n if n else 0,
        "locality": locality,
        "neg_adv_v1": neg_adv,
        "neg_adv_v2": neg_adv_v2,
        "neg_adv_v3": neg_adv_v3,
        "token_table": token_table,
        "question": question,
        "answer": answer,
        "tokens": tokens,
        "logp_cond": logp_cond,
        "logp_neg_v1": logp_neg_v1,
    }


def print_example_report(ex: dict) -> None:
    """Print one example to stdout."""
    print(f"\n{'=' * 70}")
    print(f"Task: {ex['task_id']}  row_idx={ex['row_idx']}")
    print(f"Model: {ex['model']}  strategy={ex['strategy']}")
    print(f"Tokens: {ex['n_tokens']}  mean_logp_cond={ex['mean_logp_cond']:.3f}")
    if ex["token_mismatch"]:
        print("  !! TOKEN COUNT MISMATCH — table may be misaligned !!")

    loc = ex["locality"]
    print(
        f"\nLocality (bottom 10% tokens = {loc['frac_low_tokens'] * 100:.0f}% of tokens):"
    )
    print(f"  Those tokens hold {loc['mass_frac'] * 100:.1f}% of total surprisal")
    print(
        f"  → {'LOCALIZED (>50% surprisal in top-10% tokens)' if loc['is_localized'] else 'DISTRIBUTED'}"
    )

    na = ex["neg_adv_v1"]
    print("\nNeg prompt (v1) at error tokens (z < -1.5σ):")
    print(
        f"  Error tokens: {na['n_error_tokens']} / {ex['n_tokens']} "
        f"({na['frac_error_tokens'] * 100:.1f}%)"
    )
    if not math.isnan(na["mean_delta_error"]):
        print(
            f"  mean Δ(logp_neg−logp_cond) at error tokens:  {na['mean_delta_error']:+.3f}"
        )
        print(
            f"  mean Δ(logp_neg−logp_cond) at normal tokens: {na['mean_delta_normal']:+.3f}"
        )
        print(
            f"  → Neg helps at errors: {'YES' if na['neg_helps_at_errors'] else 'NO'}"
        )

    for label, na2 in [("v2", ex["neg_adv_v2"]), ("v3", ex["neg_adv_v3"])]:
        if na2 and not math.isnan(na2["mean_delta_error"]):
            print(
                f"  neg_{label} Δ at error tokens: {na2['mean_delta_error']:+.3f}  "
                f"normal: {na2['mean_delta_normal']:+.3f}  "
                f"helps: {'YES' if na2['neg_helps_at_errors'] else 'NO'}"
            )

    print("\n--- Question (first 400 chars) ---")
    print(ex["question"][:400])
    print("\n--- Answer (first 400 chars) ---")
    print(ex["answer"][:400])
    print("\n--- Per-token table ---")
    print(ex["token_table"])


def build_markdown(all_analyses: list[dict]) -> str:
    """Build markdown report from list of analyzed examples."""
    lines = [
        "# Wrong Example Analysis — humaneval-v2.1",
        "",
        "Per-token log probability analysis of wrong examples in humaneval-v2.1.",
        "Tokenizer: gemma-4-31B-it (same tokenizer used during scoring).",
        "Scores from `humaneval_v2_1_pertok_scores.jsonl`.",
        "",
        "**Error token definition**: token with logp_cond more than 1.5σ below the",
        "completion mean. **Locality**: what fraction of total surprisal is held by the",
        "bottom-10% tokens — >50% = localized, <50% = distributed.",
        "",
        "**Neg-prompt test**: does `logp_neg_v1[i] - logp_cond[i]` (delta) average",
        "higher at error positions than at normal positions?",
        "A positive gap means the neg prompt predicts error tokens better than the",
        "positive prompt — i.e., the neg prompt is `seeing` the errors.",
        "",
    ]

    tasks_seen = []
    for ex in all_analyses:
        if ex["task_id"] not in tasks_seen:
            tasks_seen.append(ex["task_id"])
            lines.append(f"## Task: {ex['task_id']}")
            lines.append("")
            lines.append("**Question:**")
            lines.append("")
            lines.append("```python")
            lines.append(ex["question"].strip())
            lines.append("```")
            lines.append("")

        loc = ex["locality"]
        na = ex["neg_adv_v1"]
        na2 = ex.get("neg_adv_v2")
        na3 = ex.get("neg_adv_v3")

        lines.append(
            f"### {ex['task_id']} · row {ex['row_idx']} · {ex['model']} · {ex['strategy']}"
        )
        lines.append("")
        lines.append(
            f"- **Tokens**: {ex['n_tokens']}  |  **mean logp_cond**: {ex['mean_logp_cond']:.3f}"
        )
        lines.append(
            f"- **Locality**: bottom-10% tokens hold {loc['mass_frac'] * 100:.1f}% of surprisal → "
            f"{'**LOCALIZED**' if loc['is_localized'] else 'distributed'}"
        )
        if not math.isnan(na["mean_delta_error"]):
            helps_str = "**YES**" if na["neg_helps_at_errors"] else "no"
            lines.append(
                f"- **Neg v1 at error tokens** ({na['n_error_tokens']} tokens, z<−1.5σ): "
                f"Δ={na['mean_delta_error']:+.3f} vs normal Δ={na['mean_delta_normal']:+.3f} → {helps_str}"
            )
            if na2 and not math.isnan(na2["mean_delta_error"]):
                helps2 = "**YES**" if na2["neg_helps_at_errors"] else "no"
                lines.append(
                    f"- **Neg v2 at error tokens**: "
                    f"Δ={na2['mean_delta_error']:+.3f} vs normal Δ={na2['mean_delta_normal']:+.3f} → {helps2}"
                )
            if na3 and not math.isnan(na3["mean_delta_error"]):
                helps3 = "**YES**" if na3["neg_helps_at_errors"] else "no"
                lines.append(
                    f"- **Neg v3 at error tokens**: "
                    f"Δ={na3['mean_delta_error']:+.3f} vs normal Δ={na3['mean_delta_normal']:+.3f} → {helps3}"
                )
        else:
            lines.append("- No error tokens (all tokens within 1.5σ of mean)")

        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append("```python")
        lines.append(ex["answer"].strip())
        lines.append("```")
        lines.append("")
        lines.append("**Per-token table** (first 60 tokens):")
        lines.append("")
        lines.append("```")
        lines.append(ex["token_table"])
        lines.append("```")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| task | row | model | n_tok | mean_lp | localized? | n_err | neg_v1_helps? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for ex in all_analyses:
        loc = ex["locality"]
        na = ex["neg_adv_v1"]
        helps = "YES" if na["neg_helps_at_errors"] else "no"
        na_err = na["n_error_tokens"] if not math.isnan(na["mean_delta_error"]) else 0
        lines.append(
            f"| {ex['task_id']} | {ex['row_idx']} | {ex['model'].split('/')[-1][:25]} | "
            f"{ex['n_tokens']} | {ex['mean_logp_cond']:.2f} | "
            f"{'yes' if loc['is_localized'] else 'no'} | {na_err} | {helps} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/analyze_wrong_examples.py`*")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pertok",
        default="notes/log_P_diff_plots/humaneval-v2.1/humaneval_v2_1_pertok_scores.jsonl",
        help="Path to v2.1 per-token JSONL",
    )
    ap.add_argument(
        "--v21-dir",
        default="data/humaneval/v2.1",
        help="Directory containing per-task v2.1 CSVs (used for task listing only)",
    )
    ap.add_argument(
        "--v2-dir",
        default="data/humaneval/v2",
        help="Directory containing per-task v2 CSVs (source of answer text; row_idx refers to these)",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["humaneval_130", "humaneval_122"],
        help="Task IDs to analyze (default: two highest-wrong-count tasks)",
    )
    ap.add_argument(
        "--max-wrong",
        type=int,
        default=3,
        help="Max wrong examples to show per task",
    )
    ap.add_argument(
        "--output-md",
        default="notes/log_P_diff_plots/humaneval-v2.1/wrong_examples_analysis.md",
        help="Output markdown path",
    )
    ap.add_argument(
        "--tokenizer-id",
        default="google/gemma-4-31B-it",
        help="HF model ID for tokenizer (must be cached locally)",
    )
    ap.add_argument(
        "--max-table-tokens",
        type=int,
        default=60,
        help="Max tokens to show in per-token table",
    )
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.tokenizer_id}")
    tok = load_tokenizer(args.tokenizer_id)
    print(f"Tokenizer loaded. Vocab size: {tok.vocab_size}")

    print(f"Loading per-token scores from {args.pertok}")
    pertok_by_task = load_pertok_for_tasks(args.pertok, args.tasks)
    for tid in args.tasks:
        n = len(pertok_by_task[tid])
        print(f"  {tid}: {n} rows in JSONL")

    all_analyses = []

    for task_id in args.tasks:
        # row_idx in JSONL refers to v2 CSV pandas index (0-based).
        # Load v2 CSV as a dict keyed by integer index.
        v2_csv_path = os.path.join(args.v2_dir, f"{task_id}.csv")
        if not os.path.exists(v2_csv_path):
            print(
                f"  SKIP {task_id}: v2 CSV not found at {v2_csv_path}", file=sys.stderr
            )
            continue
        v2_rows = load_csv_rows(v2_csv_path)
        v2_by_idx: dict[int, dict] = {i: r for i, r in enumerate(v2_rows)}

        pertok_rows = pertok_by_task[task_id]

        # Collect wrong rows in JSONL order
        wrong_in_jsonl = [r for r in pertok_rows.values() if r["correct"] == "No"]
        wrong_in_jsonl.sort(key=lambda r: r["row_idx"])

        wrong_count = 0
        print(f"\n{'=' * 70}")
        print(
            f"TASK: {task_id}  ({len(v2_rows)} v2 rows, "
            f"{len(wrong_in_jsonl)} wrong in JSONL)"
        )

        for pertok_row in wrong_in_jsonl:
            if wrong_count >= args.max_wrong:
                break

            row_idx = pertok_row["row_idx"]
            if row_idx not in v2_by_idx:
                print(
                    f"  SKIP row_idx={row_idx}: not in v2 CSV (max idx={max(v2_by_idx)})",
                    file=sys.stderr,
                )
                continue

            csv_row = v2_by_idx[row_idx]
            ex = analyze_example(
                tok,
                csv_row,
                pertok_row,
                task_id,
                row_idx,
                max_table_tokens=args.max_table_tokens,
            )
            print_example_report(ex)
            all_analyses.append(ex)
            wrong_count += 1

    # Write markdown
    md = build_markdown(all_analyses)
    out_path = args.output_md
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\n\nMarkdown written to: {out_path}")

    # Print aggregate summary
    print(f"\n{'=' * 70}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 70}")
    localized = sum(1 for ex in all_analyses if ex["locality"]["is_localized"])
    neg_helps = sum(1 for ex in all_analyses if ex["neg_adv_v1"]["neg_helps_at_errors"])
    total = len(all_analyses)
    print(f"Examples analyzed: {total}")
    print(f"Localized errors (top-10% tokens >50% surprisal): {localized}/{total}")
    print(f"Neg prompt v1 helps at error positions: {neg_helps}/{total}")


if __name__ == "__main__":
    main()
