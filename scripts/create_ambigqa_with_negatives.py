#!/usr/bin/env python3
"""
Create train.csv and per-question test CSVs under data/ambigqa/with_negatives/.
Columns: question, answer, correct, strategy.
Gold rows: correct=Yes, strategy=ambigqa.
Negative rows: correct=No, strategy in {plausible, uninformed, clearly-wrong}.
For each question we prompt for N+3 wrong answers per strategy, filter out
matches to correct answers, then randomly sample N per strategy.
"""
import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ambigqa"
OUTPUT_DIR = DATA_DIR / "with_negatives"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "and", "but", "or", "not",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
}

PROMPT_PLAUSIBLE = '''Given this trivia question: "{question}"

The following answers are CORRECT (do not suggest these or equivalent):
{correct_list}

Generate exactly {k} answers that are PLAUSIBLE but WRONG (one per line, no numbering).'''

PROMPT_UNINFORMED = '''Given this trivia question: "{question}"

The following answers are CORRECT (do not suggest these or equivalent):
{correct_list}

Generate exactly {k} answers that an UNINFORMED person might guess (one per line, no numbering).'''

PROMPT_CLEARLY_WRONG = '''Given this trivia question: "{question}"

The following answers are CORRECT (do not suggest these or equivalent):
{correct_list}

Generate exactly {k} answers that are CLEARLY WRONG (one per line, no numbering).'''


def get_first_non_stopword(question):
    words = question.lower().replace("?", "").replace(",", "").split()
    for w in words:
        if w not in STOPWORDS:
            return w
    return words[0] if words else "unknown"


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize_answer(a):
    return " ".join(a.strip().lower().split())


def parse_answers_from_response(text):
    answers = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[\d]+[.)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        if line:
            answers.append(line.strip())
    return answers


def filter_and_sample(generated, correct_norm, n, seed):
    filtered = [a for a in generated if normalize_answer(a) not in correct_norm]
    rng = random.Random(seed)
    return rng.sample(filtered, n) if len(filtered) >= n else filtered


def query_llm(client, prompt, model):
    for attempt in range(3):
        try:
            time.sleep(0.5)
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Max retries exceeded")


def generate_negatives(client, question, correct, model, seed):
    n, k = len(correct), len(correct) + 3
    correct_norm = {normalize_answer(a) for a in correct}
    correct_list = "\n".join("- " + a for a in correct)
    out = {}
    for strategy, template in [
        ("plausible", PROMPT_PLAUSIBLE),
        ("uninformed", PROMPT_UNINFORMED),
        ("clearly-wrong", PROMPT_CLEARLY_WRONG),
    ]:
        prompt = template.format(question=question, correct_list=correct_list, k=k)
        raw = query_llm(client, prompt, model)
        parsed = parse_answers_from_response(raw)
        out[strategy] = filter_and_sample(parsed, correct_norm, n, seed)
    return out


def build_rows(question, correct, negatives):
    rows = [{"question": question, "answer": a, "correct": "Yes", "strategy": "ambigqa"} for a in correct]
    for strategy, answers in negatives.items():
        for a in answers:
            rows.append({"question": question, "answer": a, "correct": "No", "strategy": strategy})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", default=str(DATA_DIR / "train.jsonl"))
    p.add_argument("--test-jsonl", default=str(DATA_DIR / "test.jsonl"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--skip-train", action="store_true", help="Skip train, only process test")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    train_data = load_jsonl(Path(args.train_jsonl))
    test_data = load_jsonl(Path(args.test_jsonl))
    if args.limit_train:
        train_data = train_data[: args.limit_train]
    if args.limit_test:
        test_data = test_data[: args.limit_test]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question", "answer", "correct", "strategy"]

    if not args.skip_train:
        train_rows = []
        for i, item in enumerate(train_data):
            q, correct = item["question"], item["answers"]
            print(f"Train [{i+1}/{len(train_data)}] N={len(correct)}: {q[:55]}...")
            neg = generate_negatives(client, q, correct, args.model, args.seed + i)
            train_rows.extend(build_rows(q, correct, neg))
        with open(out_dir / "train.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(train_rows)
        print(f"Wrote {len(train_rows)} rows to {out_dir / 'train.csv'}")
    else:
        print("Skipping train (--skip-train)")

    for i, item in enumerate(test_data):
        q, correct = item["question"], item["answers"]
        name = get_first_non_stopword(q)
        print(f"Test [{i+1}/{len(test_data)}] {name}.csv N={len(correct)}: {q[:55]}...")
        neg = generate_negatives(client, q, correct, args.model, args.seed + 1000 + i)
        rows = build_rows(q, correct, neg)
        with open(out_dir / (name + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {len(rows)} rows to {out_dir / (name + '.csv')}")

    print("Done.")


if __name__ == "__main__":
    main()
