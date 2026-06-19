"""Split each HumanEval v2.1correct train.csv into per-problem CSVs (one per task_id).

The train/test split is problem-level and disjoint (80 train problems, 82 test problems, 0
overlap). The per-problem TEST tasks already score each test problem's own candidates; this
script produces the analogous per-problem TRAIN candidate CSVs so we can score each TRAIN
problem's own ~29 candidates (all of them; ~15 pos / ~14 neg) and average ROC over the 80
train problems -- a like-for-like comparison with the test side.

Lossless: union of the per-problem CSVs == train.csv (verified: 2283 rows, 80 problems).
Output: data/humaneval/v2.1correct-{upper,multi}-train-perproblem/humaneval_<N>.csv
(N from task_id "HumanEval/<N>"; same columns/header as the per-problem test CSVs).
Does NOT touch train.csv or any existing data — purely additive.
"""
import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIELDS = ['question', 'answer', 'correct', 'strategy', 'model', 'temperature', 'task_id', 'error']


def split(ds: str) -> None:
    src = REPO / f"data/humaneval/v2.1correct-{ds}/train.csv"
    out_dir = REPO / f"data/humaneval/v2.1correct-{ds}-train-perproblem"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src)))
    by_prob: dict[str, list[dict]] = {}
    for r in rows:
        tid = r.get("task_id", "").strip()        # "HumanEval/101"
        by_prob.setdefault(tid, []).append(r)
    written = 0
    for tid, items in by_prob.items():
        num = tid.split("/")[-1]
        out = out_dir / f"humaneval_{num}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for it in items:
                w.writerow({k: it.get(k, "") for k in FIELDS})
        written += len(items)
    assert written == len(rows), f"lossless check failed: wrote {written} != {len(rows)}"
    print(f"{ds}: {len(rows)} rows -> {len(by_prob)} per-problem CSVs in {out_dir} (lossless)")


if __name__ == "__main__":
    for ds in ["upper", "multi"]:
        split(ds)
