"""build_v2_1_correct_multi.py — the FULL humaneval-v2.1correct-multi dataset.

Drop-in replacement for v2.1: wrong rows byte-identical (locked decision 2);
every correct row stylized via libcst (comment-preserving) with a SEEDED
DIVERSE menu — one kept rename scheme + a seeded subset of kept Axis-2
transforms, deterministic by (task_id,row_idx). Menu comes from the canary's
`kept_menu.json` (canary measured ΔlogP → kept_menu → here; reproducible).

Reuses the trusted reference for problems/validate/slug (DRY). Every
transformed-correct is HumanEval-revalidated; on failure → drop Axis-2 one at
a time → rename-only → original (revert, logged). Any kept transform whose
revert rate exceeds --revert-max is AUTO-DROPPED and the build re-runs once
(bounded). Post-build assertions enforce: 100% transformed-correct pass,
wrong rows byte-identical, correct-row count preserved.

Usage:
  python build_v2_1_correct_multi.py --menu <kept_menu.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import transforms as T  # noqa: E402

_RA = _HERE.parents[1]
sys.path.insert(0, str(_RA / "scripts"))
from dataset_builder.build_humaneval_v2_1_correct_upper import (  # noqa: E402
    load_problems, slug_to_id, validate,
)

V21 = _RA / "data/humaneval/v2.1"
OUT = _RA / "data/humaneval/v2.1correct-multi"
AXIS2_PROB = 0.45  # per-kept-axis2 inclusion prob (seeded); diverse stacking


def _rng(task_id: str, row_idx: int, salt: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}|{task_id}|{row_idx}|{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _assign(task_id, row_idx, schemes, axis2, seed):
    """Deterministic per-row menu: one scheme + seeded subset of axis2."""
    s = schemes[int(_rng(task_id, row_idx, "scheme", seed) * len(schemes)) %
                len(schemes)]
    ax = tuple(a for a in axis2
               if _rng(task_id, row_idx, f"ax:{a}", seed) < AXIS2_PROB)
    return s, ax


def _stylize_with_fallback(question, answer, scheme, axis2, problem):
    """Try full menu; on validation failure progressively drop Axis-2, then
    rename-only, then original. Returns (answer, applied_scheme_bool,
    applied_axis2_list, reverted_bool, blame)."""
    attempts = [(scheme, axis2)]
    for k in range(len(axis2) - 1, -1, -1):     # drop axis2 one at a time
        attempts.append((scheme, axis2[:k]))
    attempts.append((None, ()))                  # rename-only? no: scheme-only
    if scheme is not None:
        attempts.insert(len(attempts) - 1, (scheme, ()))
    for sc, ax in attempts:
        try:
            res = T.stylize(question, answer, sc, ax)
        except Exception:  # noqa: BLE001 — treat as this attempt failing
            continue
        passed, _ = validate(problem, res["answer"])
        if passed:
            blame = ([a for a in axis2 if a not in ax] +
                     ([] if sc == scheme else (["<scheme>"] if scheme else [])))
            return (res["answer"], bool(res["meta"]["mapping"]),
                    res["meta"]["axis2_applied"], bool(blame), blame)
    return answer, False, [], True, ["<all>"]    # full revert to original


def _process(df, slug, problems, schemes, axis2, seed, stats):
    prob_fixed = None if slug == "train" else problems[slug_to_id(slug)]
    out = []
    for ri, row in df.iterrows():
        stats["total"] += 1
        is_corr = str(row.get("correct", "")).strip().lower() == "yes"
        if not is_corr:                          # wrong: byte-identical
            stats["wrong"] += 1
            out.append(row.to_dict())
            continue
        stats["correct"] += 1
        prob = (problems[row["task_id"]] if prob_fixed is None
                else prob_fixed)
        tid = slug if prob_fixed is not None else str(row["task_id"])
        sc, ax = _assign(tid, int(ri), schemes, axis2, seed)
        new_ans, sc_ok, ax_ok, rev, blame = _stylize_with_fallback(
            str(row["question"]), str(row["answer"]), sc, ax, prob)
        for b in blame:
            stats["blame"][b] = stats["blame"].get(b, 0) + 1
        for a in ax:
            stats["sel"][a] = stats["sel"].get(a, 0) + 1
        if rev:
            stats["reverted"] += 1
        nr = row.to_dict()
        nr["answer"] = new_ans
        out.append(nr)
    return pd.DataFrame(out, columns=df.columns)


def _build(schemes, axis2, seed):
    problems = load_problems()
    OUT.mkdir(parents=True, exist_ok=True)
    csvs = sorted(V21.glob("humaneval_*.csv"))
    files = csvs + ([V21 / "train.csv"] if (V21 / "train.csv").exists() else [])
    stats = dict(total=0, correct=0, wrong=0, reverted=0,
                 blame={}, sel={})
    for fp in files:
        df = pd.read_csv(fp)
        out_df = _process(df, fp.stem, problems, schemes, axis2, seed, stats)
        out_df.to_csv(OUT / fp.name, index=False)
    return stats, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", required=True, help="kept_menu.json from canary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--revert-max", type=float, default=0.10)
    args = ap.parse_args()

    menu = json.loads(Path(args.menu).read_text())
    if not menu.get("premise_supported"):
        sys.exit("ABORT: canary verdict = premise NOT supported (no kept "
                 "transforms). Per the degenerate-outcome stop, the dataset "
                 "is NOT built. Report the canary finding instead.")
    schemes = menu["kept_schemes"]
    axis2 = list(menu["kept_axis2"])
    if not schemes:
        sys.exit("ABORT: no kept rename scheme — cannot build a renamed "
                 "correct class.")

    for _pass in (1, 2):
        stats, files = _build(schemes, axis2, args.seed)
        n_corr = stats["correct"] or 1
        # auto-drop any kept Axis-2 whose blame-revert rate over selections
        # exceeds the threshold; bounded to one rebuild.
        over = [a for a in list(axis2)
                if stats["sel"].get(a, 0)
                and stats["blame"].get(a, 0) / stats["sel"][a] > args.revert_max]
        print(f"pass{_pass}: correct={stats['correct']} wrong={stats['wrong']} "
              f"reverted={stats['reverted']} "
              f"({stats['reverted']/n_corr:.1%}) over-revert={over}")
        if not over or _pass == 2:
            break
        print(f"AUTO-DROP {over} (revert>{args.revert_max:.0%}); rebuilding once")
        axis2 = [a for a in axis2 if a not in over]

    # ---- post-build assertions (fail loud) -------------------------------
    problems = load_problems()
    bad = []                 # TRANSFORMED correct rows that fail (real bug)
    pre_existing_bad = []    # rows we kept = v2.1 original that v2.1 itself
    #                          fails to validate (not ours; tolerated+reported)
    for fp in files:
        slug = fp.stem
        src = pd.read_csv(fp)
        got = pd.read_csv(OUT / fp.name)
        assert len(src) == len(got), f"row count changed in {fp.name}"
        prob_fixed = None if slug == "train" else problems[slug_to_id(slug)]
        for (_, a), (_, b) in zip(src.iterrows(), got.iterrows()):
            corr = str(a.get("correct", "")).strip().lower() == "yes"
            if not corr:
                assert str(a["answer"]) == str(b["answer"]), \
                    f"WRONG row mutated in {fp.name} (must be byte-identical)"
            else:
                prob = (problems[a["task_id"]] if prob_fixed is None
                        else prob_fixed)
                ok, _ = validate(prob, str(b["answer"]))
                if ok:
                    continue
                # A failing correct row is only OUR bug if we actually
                # transformed it. If the answer is byte-identical to the v2.1
                # original, its failure is a PRE-EXISTING v2.1 property (the
                # dataset has known correct-labeled rows that fail strict
                # re-validation) — the trusted reference builder tolerates
                # exactly this (skip_orig_fail / _validation_reverts.json). We
                # kept it untouched, so it is not ours to fix; record + report,
                # do not abort.
                if str(a["answer"]) == str(b["answer"]):
                    pre_existing_bad.append(f"{slug}:{int(a.name)}")
                else:
                    bad.append(f"{slug}:{int(a.name)}")
    assert not bad, (f"{len(bad)} TRANSFORMED correct rows FAIL HumanEval "
                     f"(real bug) — abort. e.g. {bad[:5]}")

    rep = OUT / "_BUILD_REPORT.json"
    rep.write_text(json.dumps({
        "menu": menu, "seed": args.seed, "final_axis2": axis2,
        "schemes": schemes, "stats": {k: v for k, v in stats.items()},
        "transformed_correct_failures": len(bad),  # MUST be 0
        "pre_existing_bad_v2_1_originals_kept": len(pre_existing_bad),
        "pre_existing_bad_examples": pre_existing_bad[:20],
    }, indent=2))
    print(f"OK: v2.1correct-multi built at {OUT}\n"
          f"  wrong rows byte-identical; 0 transformed-correct failures; "
          f"{len(pre_existing_bad)} pre-existing-bad v2.1 originals kept "
          f"untouched (same as v2.1correct-upper). report -> {rep}")


if __name__ == "__main__":
    main()
