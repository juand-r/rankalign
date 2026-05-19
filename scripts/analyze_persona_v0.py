"""Basic EDA for the persona-v0 dataset.

Produces, per persona and pooled:
  - statement length distribution (chars + GPT-2 tokens), split by correct=yes/no
  - label balance (sanity)
  - label_confidence distribution
  - cohen's d for the (yes - no) length difference, in tokens (this is the
    "length leakage" we worry about: if persona-matching statements are
    systematically longer/shorter, raw gen-score picks up that signal instead
    of persona content; TC's job is to remove exactly that)

Outputs:
  results/persona_v0_eda/summary.csv
  results/persona_v0_eda/length_hist_<persona>.png
  results/persona_v0_eda/length_box_pooled.png
  results/persona_v0_eda/label_confidence_hist.png
  results/persona_v0_eda/length_vs_label_per_persona.png

Run:
  source /u/jdr/venvs/venv_lexcons/bin/activate
  python scripts/analyze_persona_v0.py
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "persona" / "v0"
OUT_DIR = ROOT / "results" / "persona_v0_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _gpt2_token_count_factory():
    """Lazy GPT-2 tokenizer with a regex fallback if unavailable."""
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")

        def count(s: str) -> int:
            return len(tok.encode(s))

        return count, "gpt2"
    except Exception as e:
        print(f"[warn] GPT-2 tokenizer unavailable ({e}); falling back to whitespace+punct split")
        import re
        token_re = re.compile(r"\w+|[^\w\s]")

        def count(s: str) -> int:
            return len(token_re.findall(s))

        return count, "regex"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"persona-v0 data dir not found: {DATA_DIR}")

    count_tokens, tok_kind = _gpt2_token_count_factory()
    print(f"Tokenizer: {tok_kind}")

    files = sorted(p for p in DATA_DIR.glob("persona-*.csv"))
    train_csv = DATA_DIR / "train.csv"

    summary_rows: list[dict] = []

    pooled_yes_tok: list[int] = []
    pooled_no_tok: list[int] = []
    pooled_yes_chars: list[int] = []
    pooled_no_chars: list[int] = []

    persona_panels: list[tuple[str, np.ndarray, np.ndarray]] = []  # (persona, yes_tok, no_tok)

    label_conf_by_persona: OrderedDict[str, list[float]] = OrderedDict()

    for path in files:
        persona = path.stem.removeprefix("persona-")
        rows = _load_csv(path)
        chars = np.array([len(r["statement"]) for r in rows])
        toks = np.array([count_tokens(r["statement"]) for r in rows])
        label_yes = np.array([r["correct"] == "yes" for r in rows])
        label_conf = np.array([float(r["label_confidence"]) for r in rows])

        yes_tok = toks[label_yes]
        no_tok = toks[~label_yes]
        yes_chars = chars[label_yes]
        no_chars = chars[~label_yes]

        pooled_yes_tok.extend(yes_tok.tolist())
        pooled_no_tok.extend(no_tok.tolist())
        pooled_yes_chars.extend(yes_chars.tolist())
        pooled_no_chars.extend(no_chars.tolist())

        persona_panels.append((persona, yes_tok, no_tok))
        label_conf_by_persona[persona] = label_conf.tolist()

        summary_rows.append({
            "persona": persona,
            "n": len(rows),
            "n_yes": int(label_yes.sum()),
            "n_no": int((~label_yes).sum()),
            "tok_mean_yes": round(float(yes_tok.mean()), 2),
            "tok_mean_no": round(float(no_tok.mean()), 2),
            "tok_diff_yes_minus_no": round(float(yes_tok.mean() - no_tok.mean()), 2),
            "tok_cohen_d": round(float(_cohens_d(yes_tok, no_tok)), 3),
            "char_mean_yes": round(float(yes_chars.mean()), 1),
            "char_mean_no": round(float(no_chars.mean()), 1),
            "label_conf_mean": round(float(label_conf.mean()), 3),
            "label_conf_min": round(float(label_conf.min()), 3),
            "label_conf_p10": round(float(np.percentile(label_conf, 10)), 3),
            "label_conf_max": round(float(label_conf.max()), 3),
        })

    # train.csv aggregate row
    rows_train = _load_csv(train_csv)
    chars = np.array([len(r["statement"]) for r in rows_train])
    toks = np.array([count_tokens(r["statement"]) for r in rows_train])
    label_yes = np.array([r["correct"] == "yes" for r in rows_train])
    yes_tok = toks[label_yes]
    no_tok = toks[~label_yes]

    summary_rows.append({
        "persona": "_TRAIN_POOLED",
        "n": len(rows_train),
        "n_yes": int(label_yes.sum()),
        "n_no": int((~label_yes).sum()),
        "tok_mean_yes": round(float(yes_tok.mean()), 2),
        "tok_mean_no": round(float(no_tok.mean()), 2),
        "tok_diff_yes_minus_no": round(float(yes_tok.mean() - no_tok.mean()), 2),
        "tok_cohen_d": round(float(_cohens_d(yes_tok, no_tok)), 3),
        "char_mean_yes": round(float(chars[label_yes].mean()), 1),
        "char_mean_no": round(float(chars[~label_yes].mean()), 1),
        "label_conf_mean": "",
        "label_conf_min": "",
        "label_conf_p10": "",
        "label_conf_max": "",
    })

    # --- write summary CSV ---
    summary_path = OUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"Wrote {summary_path}")

    # --- per-persona length histograms (yes vs no, in tokens) ---
    n = len(persona_panels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    pooled_max = max(max(yes.max(), no.max()) for _, yes, no in persona_panels)
    bins = np.arange(0, pooled_max + 2, 1)
    for ax, (persona, yes_tok, no_tok) in zip(axes.flat, persona_panels):
        ax.hist(no_tok, bins=bins, alpha=0.55, label=f"no  (n={len(no_tok)})", color="#d62728")
        ax.hist(yes_tok, bins=bins, alpha=0.55, label=f"yes (n={len(yes_tok)})", color="#2ca02c")
        d = _cohens_d(yes_tok, no_tok)
        ax.set_title(f"{persona}\nΔmean={yes_tok.mean()-no_tok.mean():+.2f} tok, d={d:+.2f}", fontsize=9)
        ax.set_xlabel("tokens (gpt2)")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Statement length by persona-match label (yes = matches persona)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "length_hist_per_persona.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- pooled boxplot ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot([pooled_no_tok, pooled_yes_tok], labels=["no (anti-persona)", "yes (matches persona)"], showmeans=True)
    pooled_d = _cohens_d(np.array(pooled_yes_tok), np.array(pooled_no_tok))
    ax.set_ylabel("tokens (gpt2)")
    ax.set_title(f"Pooled length by label (across all 8 personas, n={len(pooled_yes_tok)+len(pooled_no_tok)})\n"
                 f"Δmean = {np.mean(pooled_yes_tok)-np.mean(pooled_no_tok):+.2f}, "
                 f"Cohen d = {pooled_d:+.3f}")
    fig.tight_layout()
    out = OUT_DIR / "length_box_pooled.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- label confidence overlay ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for persona, vals in label_conf_by_persona.items():
        ax.hist(vals, bins=np.linspace(0.4, 1.0, 30), alpha=0.45, label=persona)
    ax.set_xlabel("label_confidence (per source dataset)")
    ax.set_ylabel("count")
    ax.set_title("label_confidence distribution by persona")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "label_confidence_hist.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- length vs label scatter (mean ± std) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(persona_panels))
    yes_means = [yes.mean() for _, yes, _ in persona_panels]
    yes_stds = [yes.std(ddof=1) for _, yes, _ in persona_panels]
    no_means = [no.mean() for _, _, no in persona_panels]
    no_stds = [no.std(ddof=1) for _, _, no in persona_panels]
    ax.errorbar(xs - 0.12, yes_means, yerr=yes_stds, fmt="o", color="#2ca02c", label="yes (matches persona)", capsize=3)
    ax.errorbar(xs + 0.12, no_means, yerr=no_stds, fmt="o", color="#d62728", label="no (anti-persona)", capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([p for p, _, _ in persona_panels], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("tokens (gpt2), mean ± SD")
    ax.set_title("Mean statement length by label, per persona")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "length_vs_label_per_persona.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- print summary inline ---
    print("\n=== SUMMARY ===")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
