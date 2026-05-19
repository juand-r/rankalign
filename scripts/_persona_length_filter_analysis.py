"""How much data do we lose, and does the length confound shrink, if we cap
statement length at 20 / 25 GPT-2 tokens? Also report char-equivalent cutoffs.

Run:
  source /u/jdr/venvs/venv_lexcons/bin/activate
  python scripts/_persona_length_filter_analysis.py
"""
from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "persona" / "v0"

tok = GPT2TokenizerFast.from_pretrained("gpt2")


def load(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float("nan") if pooled == 0 else (a.mean() - b.mean()) / pooled


def percentile_by_n(arr: np.ndarray, n: int) -> float:
    return float(np.percentile(arr, n))


# Load every CSV in v0 and build a single dataframe-like list of dicts.
all_rows: list[dict] = []
sources: list[tuple[str, Path]] = [("_train", DATA_DIR / "train.csv")]
for p in sorted(DATA_DIR.glob("persona-*.csv")):
    sources.append((p.stem.removeprefix("persona-"), p))

print("Counting tokens for every CSV...")
file_summary = []
for label, path in sources:
    rows = load(path)
    chars = np.array([len(r["statement"]) for r in rows])
    toks = np.array([len(tok.encode(r["statement"])) for r in rows])
    yes = np.array([r["correct"] == "yes" for r in rows])
    for i, r in enumerate(rows):
        all_rows.append({
            "src_label": label,
            "src_path": str(path.relative_to(ROOT)),
            "persona": r["persona"],
            "correct": r["correct"],
            "statement": r["statement"],
            "n_chars": int(chars[i]),
            "n_tok": int(toks[i]),
            "yes": bool(yes[i]),
        })
    file_summary.append((label, len(rows), float(chars.mean()), float(toks.mean()),
                         float(chars.mean() / toks.mean())))

print("\n=== chars / token by file ===")
print(f"{'file':40} {'n':>6} {'mean_chars':>11} {'mean_tok':>9} {'chars/tok':>10}")
for label, n, mc, mt, cpt in file_summary:
    print(f"{label:40} {n:>6} {mc:>11.2f} {mt:>9.2f} {cpt:>10.3f}")

# Pool all rows (note: train.csv is a copy of train halves of in-domain persona
# files, so it overlaps with persona-*.csv. For "global stats" we use the
# de-duplicated union of the persona-*.csv files only — i.e. drop the train.csv
# entries.)
no_train = [r for r in all_rows if r["src_label"] != "_train"]
n_chars = np.array([r["n_chars"] for r in no_train])
n_tok = np.array([r["n_tok"] for r in no_train])

print(f"\nGlobal (8 persona-*.csv files, n={len(no_train)}):")
print(f"  mean chars/token = {n_chars.mean()/n_tok.mean():.3f}")
print(f"  median chars/token (per row) = {np.median(n_chars / n_tok):.3f}")

# What char threshold matches 20 / 25 tokens?
# Method 1: mean ratio.
ratio_mean = n_chars.mean() / n_tok.mean()
# Method 2: char p such that n_chars <= p captures the same rows as n_tok <= K.
for K in (20, 25):
    keep = n_tok <= K
    n_kept = int(keep.sum())
    char_max_among_kept = int(n_chars[keep].max())
    print(f"\n--- token cap = {K} ---")
    print(f"  rows kept (n_tok <= {K}): {n_kept}/{len(no_train)} ({100*n_kept/len(no_train):.1f}%)")
    print(f"  rows lost: {len(no_train)-n_kept} ({100*(1-n_kept/len(no_train)):.1f}%)")
    print(f"  char-equivalent (max chars among kept rows): {char_max_among_kept}")
    print(f"  char-equivalent via mean ratio ({ratio_mean:.2f} chars/tok): ~{round(K*ratio_mean)}")

# Now: per-persona length-by-label after capping at 20 and 25 tokens.
print("\n\n=== length confound after token cap ===")
for K in (20, 25):
    print(f"\n--- per-persona Cohen's d AFTER token cap K={K} ---")
    print(f"{'persona':35} {'n_kept':>7} {'pct_kept':>9} {'mean_yes':>9} {'mean_no':>8} {'Δ':>7} {'d':>7}")
    persona_groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in no_train:
        persona_groups.setdefault(r["persona"], []).append(r)
    for persona, rows in persona_groups.items():
        kept = [r for r in rows if r["n_tok"] <= K]
        if len(kept) < 4:
            continue
        toks_yes = np.array([r["n_tok"] for r in kept if r["yes"]])
        toks_no = np.array([r["n_tok"] for r in kept if not r["yes"]])
        d = cohens_d(toks_yes, toks_no)
        print(f"{persona:35} {len(kept):>7} {100*len(kept)/len(rows):>8.1f}% "
              f"{toks_yes.mean():>9.2f} {toks_no.mean():>8.2f} "
              f"{toks_yes.mean()-toks_no.mean():>+7.2f} {d:>+7.3f}")
    # Pooled
    kept = [r for r in no_train if r["n_tok"] <= K]
    toks_yes = np.array([r["n_tok"] for r in kept if r["yes"]])
    toks_no = np.array([r["n_tok"] for r in kept if not r["yes"]])
    d = cohens_d(toks_yes, toks_no)
    print(f"{'POOLED':35} {len(kept):>7} {100*len(kept)/len(no_train):>8.1f}% "
          f"{toks_yes.mean():>9.2f} {toks_no.mean():>8.2f} "
          f"{toks_yes.mean()-toks_no.mean():>+7.2f} {d:>+7.3f}")

# Also report data loss specifically for train.csv (which is the one we'd
# actually shrink for training) and for each test file.
print("\n\n=== per-file data loss at token caps ===")
print(f"{'file':40} {'n':>6} {'kept@20':>8} {'pct@20':>7} {'kept@25':>8} {'pct@25':>7}")
for label, path in sources:
    rows_here = [r for r in all_rows if r["src_label"] == label]
    n = len(rows_here)
    n20 = sum(1 for r in rows_here if r["n_tok"] <= 20)
    n25 = sum(1 for r in rows_here if r["n_tok"] <= 25)
    print(f"{label:40} {n:>6} {n20:>8} {100*n20/n:>6.1f}% {n25:>8} {100*n25/n:>6.1f}%")

# Length distribution: pXX percentiles of n_tok across the pooled non-train data
print("\n\n=== global token-length percentiles (n=8000, 8 persona-*.csv files) ===")
for p in (50, 75, 90, 95, 97.5, 99, 99.5):
    print(f"  p{p:>4} = {np.percentile(n_tok, p):.1f} tokens")

# And for the train pool only (n=2500)
train_rows = [r for r in all_rows if r["src_label"] == "_train"]
toks_train = np.array([r["n_tok"] for r in train_rows])
print(f"\n=== train.csv token-length percentiles (n={len(train_rows)}) ===")
for p in (50, 75, 90, 95, 97.5, 99, 99.5):
    print(f"  p{p:>4} = {np.percentile(toks_train, p):.1f} tokens")
