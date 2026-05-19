"""Build persona-v0 dataset from the Anthropic / Perez et al. persona JSONL files.

Source data: data/persona/raw/{<persona_slug>.jsonl, ...}
Output:      data/persona/v0/{train.csv, persona-<slug>.csv, _BUILD_REPORT.json}

Split shape (locked decision, 2026-05-18):
    5 in-domain personas, 3 cross-persona held-out personas (Option C / cross-cluster).
    In-domain personas are split 50/50 within-persona (deterministic, seed=0). The
    train halves are pooled into train.csv; the test halves go to per-persona
    persona-<slug>.csv files. Held-out personas keep all 1000 rows in their
    persona-<slug>.csv (cross-persona OOD eval).

Persona assignments:
    in-domain (train + within-persona test):
        psychopathy, machiavellianism, narcissism,
        subscribes-to-moral-nihilism, believes-life-has-no-meaning
    held-out (cross-persona test only):
        desire-to-create-allies, interest-in-music, interest-in-science

Row schema (CSV columns):
    persona            persona slug (e.g. "psychopathy")
    statement          the statement text
    correct            "yes" iff this statement matches the persona direction
                       (i.e. answer_matching_behavior == " Yes")
    label_confidence   float in [0,1] from the source JSONL

Run:
    source /u/jdr/venvs/venv_lexcons/bin/activate
    python scripts/dataset_builder/build_persona_v0.py
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
OUT_DIR = os.path.join(REPO_ROOT, "data", "persona", "v0")

IN_DOMAIN_PERSONAS = [
    "psychopathy",
    "machiavellianism",
    "narcissism",
    "subscribes-to-moral-nihilism",
    "believes-life-has-no-meaning",
]

HELD_OUT_PERSONAS = [
    "desire-to-create-allies",
    "interest-in-music",
    "interest-in-science",
]

CSV_HEADER = ["persona", "statement", "correct", "label_confidence"]
SEED = 0


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_from_raw(persona: str, raw: dict) -> dict:
    """Convert a single raw JSONL record into our CSV row dict."""
    matches = raw["answer_matching_behavior"].strip() == "Yes"
    return {
        "persona": persona,
        "statement": raw["statement"],
        "correct": "yes" if matches else "no",
        "label_confidence": float(raw["label_confidence"]),
    }


def _split_50_50(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified 50/50 within-persona split, balanced by `correct`.

    Stratification matters because each persona has exactly 500 yes / 500 no;
    a non-stratified shuffle could land on a 47/53 split by chance, which would
    drift per-task AUC noise unnecessarily.
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
    report["split_shape"] = "5 in-domain (50/50 within-persona) + 3 held-out (cross-persona)"
    report["in_domain_personas"] = IN_DOMAIN_PERSONAS
    report["held_out_personas"] = HELD_OUT_PERSONAS
    report["personas"] = OrderedDict()

    train_pool: list[dict] = []

    # in-domain: 50/50 within-persona; train half feeds the pool
    for persona in IN_DOMAIN_PERSONAS:
        raw_path = os.path.join(args.raw_dir, f"{persona}.jsonl")
        if not os.path.exists(raw_path):
            raise SystemExit(f"missing raw JSONL for {persona}: {raw_path}")
        rows = [_row_from_raw(persona, r) for r in _load_jsonl(raw_path)]
        train_rows, test_rows = _split_50_50(rows, args.seed)
        train_pool.extend(train_rows)
        if not args.dry_run:
            _write_csv(os.path.join(args.out_dir, f"persona-{persona}.csv"), test_rows)
        report["personas"][persona] = {
            "kind": "in-domain",
            "raw": _label_balance(rows),
            "train_half": _label_balance(train_rows),
            "test_half": _label_balance(test_rows),
        }

    # shuffle pooled train.csv with the same RNG (seeded distinctly so it's deterministic)
    rng = random.Random(args.seed + 1)
    rng.shuffle(train_pool)
    if not args.dry_run:
        _write_csv(os.path.join(args.out_dir, "train.csv"), train_pool)
    report["train_csv"] = _label_balance(train_pool)

    # held-out: full file goes to test
    for persona in HELD_OUT_PERSONAS:
        raw_path = os.path.join(args.raw_dir, f"{persona}.jsonl")
        if not os.path.exists(raw_path):
            raise SystemExit(f"missing raw JSONL for {persona}: {raw_path}")
        rows = [_row_from_raw(persona, r) for r in _load_jsonl(raw_path)]
        if not args.dry_run:
            _write_csv(os.path.join(args.out_dir, f"persona-{persona}.csv"), rows)
        report["personas"][persona] = {
            "kind": "held-out",
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
