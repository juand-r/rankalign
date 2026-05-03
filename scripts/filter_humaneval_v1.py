"""Filter humaneval-v1 data: keep only answers with lines < 30 AND chars < 1250.

Overwrites CSVs in data/humaneval/v1/ in place.
"""

import glob
import sys
from pathlib import Path

import pandas as pd

MAX_LINES = 30
MAX_CHARS = 1250

DATA_DIR = Path("data/humaneval/v1")


def passes_filter(answer: str) -> bool:
    n_chars = len(answer)
    n_lines = answer.count("\n") + 1
    return n_lines < MAX_LINES and n_chars < MAX_CHARS


def filter_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    n_before = len(df)
    mask = df["answer"].apply(passes_filter)
    df_filtered = df[mask]
    n_after = len(df_filtered)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        df_filtered.to_csv(path, index=False)

    return {
        "file": path.name,
        "before": n_before,
        "after": n_after,
        "dropped": n_dropped,
    }


def main():
    task_files = sorted(DATA_DIR.glob("humaneval_*.csv"))
    train_file = DATA_DIR / "train.csv"

    all_files = task_files + ([train_file] if train_file.exists() else [])
    print(f"Filtering {len(all_files)} files (lines < {MAX_LINES}, chars < {MAX_CHARS})")
    print("-" * 60)

    total_before = total_after = 0
    for f in all_files:
        result = filter_csv(f)
        total_before += result["before"]
        total_after += result["after"]
        if result["dropped"] > 0:
            print(f"  {result['file']:35s}  {result['before']:4d} → {result['after']:4d}  "
                  f"(dropped {result['dropped']})")

    total_dropped = total_before - total_after
    print("-" * 60)
    print(f"Total: {total_before} → {total_after}  (dropped {total_dropped}, "
          f"kept {total_after/total_before*100:.1f}%)")


if __name__ == "__main__":
    main()
