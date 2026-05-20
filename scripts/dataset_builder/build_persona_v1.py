"""Build persona-v1 dataset from the Anthropic / Perez et al. persona JSONL files.

Differences from persona-v0 (locked decision, 2026-05-20):
  1) Two in-domain personas dropped because the validator on base gemma-2-*-it
     could not separate yes/no on them (val ROC-AUC ~50, val accuracy floor),
     making them a poor v2g testbed (both validator and generator carry no
     useful signal):
        - subscribes-to-moral-nihilism
        - believes-life-has-no-meaning
     (See the "Validator ROC-AUC (eval = self/neg-typcorr)" 0.Base rows for
      gemma-2-9b-it / 2b-it in output-metrics/persona_v0_tables.md.)
  2) Labels FLIPPED for the 3 antisocial in-domain personas, so that
     `correct = "yes"` is the *prosocial* / "more positive" direction in every
     persona used by v1:
        - psychopathy            (yes  -> "not psychopath")
        - machiavellianism       (yes  -> "not machiavellic")
        - narcissism             (yes  -> "not narcissistic")
     The 3 OOD held-out personas (desire-to-create-allies, interest-in-music,
     interest-in-science) are kept as-is because their `yes` direction is
     already the positive one.

  Everything else is identical to v0: same raw JSONL files, same row schema
  (persona, statement, correct, label_confidence), same 50/50 within-persona
  stratified train/test split (seed=0), same train.csv pooling for ID train
  halves, same per-persona-<slug>.csv eval files. OOD held-out personas keep
  all 1000 rows in the per-persona file.

Source data: data/persona/raw/{<persona_slug>.jsonl, ...}
Output:      data/persona/v1/{train.csv, persona-<slug>.csv, _BUILD_REPORT.json}

Persona assignments (v1):
    in-domain (train + within-persona test, labels FLIPPED):
        psychopathy, machiavellianism, narcissism
    held-out (cross-persona test only, labels UNCHANGED):
        desire-to-create-allies, interest-in-music, interest-in-science

Row schema (CSV columns):
    persona            persona slug (e.g. "psychopathy")
    statement          the statement text (unchanged from v0/raw)
    correct            "yes" iff the v1 positive direction holds
                       (= "No" in raw JSONL for the 3 flipped antisocial
                        personas, = "Yes" for the 3 OOD personas)
    label_confidence   float in [0,1] from the source JSONL (unchanged)

Run:
    source /u/jdr/venvs/venv_lexcons/bin/activate
    python scripts/dataset_builder/build_persona_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import OrderedDict
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, "data", "persona", "raw")
OUT_DIR = os.path.join(REPO_ROOT, "data", "persona", "v1")

# In-domain personas in v1: ONLY the 3 antisocial ones. Labels flipped.
IN_DOMAIN_PERSONAS = [
    "psychopathy",
    "machiavellianism",
    "narcissism",
]

# Held-out personas in v1: same 3 as v0. Labels NOT flipped.
HELD_OUT_PERSONAS = [
    "desire-to-create-allies",
    "interest-in-music",
    "interest-in-science",
]

# These personas have their "correct" column flipped (yes <-> no) so that the
# positive direction (yes) is the prosocial side rather than the persona side.
FLIP_LABEL_PERSONAS = {
    "psychopathy",
    "machiavellianism",
    "narcissism",
}

# Personas dropped from v0 (kept here as documentation of what's missing in v1).
DROPPED_FROM_V0 = [
    "subscribes-to-moral-nihilism",
    "believes-life-has-no-meaning",
]

CSV_HEADER = ["persona", "statement", "correct", "label_confidence"]
SEED = 0


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_from_raw(persona: str, raw: dict, flip: bool) -> dict:
    """Convert a single raw JSONL record into our CSV row dict.

    `flip=True` swaps the yes/no label so that "yes" corresponds to
    answer_matching_behavior == " No" (used for the 3 antisocial personas).
    """
    matches = raw["answer_matching_behavior"].strip() == "Yes"
    if flip:
        matches = not matches
    return {
        "persona": persona,
        "statement": raw["statement"],
        "correct": "yes" if matches else "no",
        "label_confidence": float(raw["label_confidence"]),
    }


def _split_50_50(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified 50/50 within-persona split, balanced by `correct`.

    Each raw persona file has exactly 500 yes / 500 no in the original (Perez)
    encoding. After a label flip the 50/50 balance is preserved (just relabeled),
    so stratifying on the post-flip `correct` still yields a clean 50/50 split.
    """
    rng = random.Random(seed)
    pos = [r for r in rows if r["correct"] == "yes"]
    neg = [r for r in rows if r["correct"] == "no"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half_pos = len(pos) // 2
    half_neg = len(neg) // 2
    train = pos[:half_pos] + neg[:half_neg]
    test = pos[half_pos:] + neg[half_neg:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_HEADER})


def _label_balance(rows: list[dict]) -> dict:
    pos = sum(1 for r in rows if r["correct"] == "yes")
    return {"n": len(rows), "yes": pos, "no": len(rows) - pos}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        raise SystemExit(f"raw dir not found: {args.raw_dir}")

    report = OrderedDict()
    report["built_at"] = datetime.utcnow().isoformat() + "Z"
    report["raw_dir"] = os.path.relpath(args.raw_dir, REPO_ROOT)
    report["out_dir"] = os.path.relpath(args.out_dir, REPO_ROOT)
    report["seed"] = args.seed
    report["split_shape"] = "3 in-domain (50/50 within-persona, labels flipped) + 3 held-out (cross-persona, labels unchanged)"
    report["in_domain_personas"] = IN_DOMAIN_PERSONAS
    report["held_out_personas"] = HELD_OUT_PERSONAS
    report["flip_label_personas"] = sorted(FLIP_LABEL_PERSONAS)
    report["dropped_from_v0"] = DROPPED_FROM_V0
    report["personas"] = OrderedDict()

    train_pool: list[dict] = []

    # in-domain: 50/50 within-persona; train half feeds the pool. Labels flipped
    # iff the persona is in FLIP_LABEL_PERSONAS.
    for persona in IN_DOMAIN_PERSONAS:
        raw_path = os.path.join(args.raw_dir, f"{persona}.jsonl")
        if not os.path.exists(raw_path):
            raise SystemExit(f"missing raw JSONL for {persona}: {raw_path}")
        flip = persona in FLIP_LABEL_PERSONAS
        rows = [_row_from_raw(persona, r, flip=flip) for r in _load_jsonl(raw_path)]
        train_rows, test_rows = _split_50_50(rows, args.seed)
        train_pool.extend(train_rows)
        if not args.dry_run:
            _write_csv(os.path.join(args.out_dir, f"persona-{persona}.csv"), test_rows)
        report["personas"][persona] = {
            "kind": "in-domain",
            "label_flipped": flip,
            "raw": _label_balance(rows),
            "train_half": _label_balance(train_rows),
            "test_half": _label_balance(test_rows),
        }

    rng = random.Random(args.seed + 1)
    rng.shuffle(train_pool)
    if not args.dry_run:
        _write_csv(os.path.join(args.out_dir, "train.csv"), train_pool)
    report["train_csv"] = _label_balance(train_pool)

    # held-out: full file goes to test. Labels NOT flipped.
    for persona in HELD_OUT_PERSONAS:
        raw_path = os.path.join(args.raw_dir, f"{persona}.jsonl")
        if not os.path.exists(raw_path):
            raise SystemExit(f"missing raw JSONL for {persona}: {raw_path}")
        flip = persona in FLIP_LABEL_PERSONAS  # always False here, but explicit
        rows = [_row_from_raw(persona, r, flip=flip) for r in _load_jsonl(raw_path)]
        if not args.dry_run:
            _write_csv(os.path.join(args.out_dir, f"persona-{persona}.csv"), rows)
        report["personas"][persona] = {
            "kind": "held-out",
            "label_flipped": flip,
            "raw": _label_balance(rows),
            "test_all": _label_balance(rows),
        }

    if not args.dry_run:
        report_path = os.path.join(args.out_dir, "_BUILD_REPORT.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
