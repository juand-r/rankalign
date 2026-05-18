"""build_canary.py — emit canary_pairs.jsonl for the correct-multi ΔlogP
canary. CPU only (libcst + HumanEval unit-test validation). No GPU, no model.

Per sampled correct v2.1 row, emit variants:
  original | noop(libcst round-trip) | each rename scheme alone |
  each Axis-2 transform alone (scheme=None) | a few stacked combos |
  one quarantined negative-control (MUST fail validation).

Every variant is HumanEval-revalidated via the trusted reference `validate`.
Schema (one line per VARIANT, incl. original — no info.mapping gate):
  {task_id,row_idx,question,variant_id,label,scheme,axis2,negative_control,
   answer,validated,revert_reason}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import transforms as T  # noqa: E402

_RANKALIGN = _HERE.parents[1]
sys.path.insert(0, str(_RANKALIGN / "scripts"))
from dataset_builder.build_humaneval_v2_1_correct_upper import (  # noqa: E402
    load_problems, slug_to_id, validate,
)

STACKED = [
    ("upper", ("redundant_temp", "inject_comment")),
    ("cryptic", ("boolean_expand",)),
    ("verbose", ("dead_cruft", "redundant_temp")),
]


def _variants(question, answer):
    """Yield (label, scheme, axis2, neg) specs."""
    yield ("original", "__orig__", (), False)
    yield ("noop", None, (), False)
    for s in T.RENAME_SCHEMES:
        yield (f"scheme:{s}", s, (), False)
    for t in T.AXIS2:
        yield (f"axis2:{t}", None, (t,), False)
    for i, (s, ax) in enumerate(STACKED):
        yield (f"stacked:{i}:{s}+{'+'.join(ax)}", s, ax, False)
    yield ("neg_control", None, (), True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v21-dir", default=str(_RANKALIGN / "data/humaneval/v2.1"))
    ap.add_argument("--out", default=str(_HERE.parents[3] /
                    "notes/log_P_diff_plots/humaneval-v2.1correct-multi/canary_pairs.jsonl"))
    ap.add_argument("--n-rows", type=int, default=36)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    problems = load_problems()
    csvs = sorted(Path(args.v21_dir).glob("humaneval_*.csv"))
    rng.shuffle(csvs)

    picked = []  # (slug, row_idx, question, answer)
    per_task = max(1, -(-args.n_rows // args.n_tasks))  # ceil
    for csv in csvs:
        if len({p[0] for p in picked}) >= args.n_tasks and len(picked) >= args.n_rows:
            break
        slug = csv.stem
        if slug_to_id(slug) not in problems:
            continue
        df = pd.read_csv(csv)
        cr = df[df["correct"].astype(str).str.lower() == "yes"]
        if len(cr) == 0:
            continue
        cr = cr.assign(_len=cr["answer"].astype(str).str.len())
        # ≥1 long body per task: always take the longest, then random others
        longest = cr.loc[cr["_len"].idxmax()]
        rest = cr.drop(longest.name)
        take = [longest] + (
            [rest.loc[i] for i in rng.sample(list(rest.index),
             min(per_task - 1, len(rest)))] if len(rest) else [])
        for row in take:
            picked.append((slug, int(row.name), str(row["question"]),
                           str(row["answer"])))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_var = n_rev = 0
    with open(args.out, "w") as f:
        for slug, ridx, question, answer in picked:
            problem = problems[slug_to_id(slug)]
            for label, scheme, axis2, neg in _variants(question, answer):
                rec = {"task_id": slug, "row_idx": ridx, "question": question,
                       "variant_id": f"{slug}:{ridx}:{label}", "label": label,
                       "scheme": scheme, "axis2": list(axis2),
                       "negative_control": neg,
                       # what ACTUALLY applied (from stylize meta) — analyze
                       # aggregates a unit only over rows where it applied, so
                       # structural no-ops don't dilute the keep/cut stats.
                       "scheme_applied": False, "axis2_applied": []}
                try:
                    if label == "original":
                        # Raw dataset row — kept for provenance only. NOT the
                        # ΔlogP baseline: the `noop` variant is (it goes
                        # through the identical build_full_func_src+to_v2_format
                        # pipeline, so any re-indent of column-0 trailing code
                        # is COMMON to baseline+variants and cancels in Δ). We
                        # do not re-judge dataset labels (reference builder's
                        # skip_orig_fail path); validate() backs *transforms*.
                        rec.update(answer=answer, validated=True,
                                   revert_reason=None)
                        n_var += 1
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    res = T.stylize(question, answer, scheme, axis2,
                                    negative_control=neg)
                    out_ans = res["answer"]
                    rec["scheme_applied"] = bool(res["meta"]["mapping"])
                    rec["axis2_applied"] = list(res["meta"]["axis2_applied"])
                    passed, err = validate(problem, out_ans)
                    rec.update(answer=out_ans, validated=bool(passed),
                               revert_reason=(None if passed else (err or "")[:300]))
                except Exception as e:  # noqa: BLE001 — record, do not crash the build
                    rec.update(answer=None, validated=False,
                               revert_reason=f"{type(e).__name__}: {e}"[:300])
                if not rec["validated"]:
                    n_rev += 1
                n_var += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_rows = len(picked)
    n_tasks = len({p[0] for p in picked})
    print(f"rows={n_rows} tasks={n_tasks} variants={n_var} reverts={n_rev}")
    print(f"-> {args.out}")
    # Sanity the canary itself depends on:
    assert n_rows >= args.n_rows, f"only {n_rows} rows (< {args.n_rows})"
    assert n_tasks >= args.n_tasks, f"only {n_tasks} tasks (< {args.n_tasks})"


if __name__ == "__main__":
    main()
