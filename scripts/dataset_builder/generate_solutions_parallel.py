#!/usr/bin/env python3
"""
Parallel solution generator for vLLM backends.

Sends concurrent requests to maximize GPU utilization on vLLM servers.
Uses the same prompt/validation logic as generate_solutions.py.

Usage:
    python scripts/dataset_builder/generate_solutions_parallel.py \
        --problems data/humaneval/problems.jsonl \
        --output data/humaneval/solutions.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --base-url http://localhost:8000/v1 \
        --samples 60 --workers 30 \
        --filter-ids "HumanEval/2,HumanEval/3,..."
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from solution_generator.humaneval_config import HumanEvalConfig


def generate_one(client, model, problem, task_config, temperature, max_tokens, strategy="normal"):
    """Generate and validate a single solution. Returns a record dict or None on error."""
    try:
        messages = task_config.make_prompt(problem, strategy=strategy)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        solution = task_config.clean_solution(raw)
        passed, extracted_answer, error = task_config.validate(problem, solution)

        return task_config.build_record(
            problem=problem,
            solution=solution,
            raw_solution=raw,
            passed=passed,
            extracted_answer=extracted_answer,
            error=error,
            temperature=temperature,
            model=model,
            strategy=strategy,
        )
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Parallel solution generator for vLLM")
    parser.add_argument("--problems", required=True, help="problems.jsonl path")
    parser.add_argument("--output", required=True, help="Output JSONL (appends)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="vLLM server URL")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--samples", type=int, default=60, help="Samples per problem")
    parser.add_argument("--workers", type=int, default=30, help="Concurrent requests")
    parser.add_argument("--temps", default="0.7,1.0,1.2", help="Temperature schedule")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--filter-ids", help="Comma-separated task IDs")
    parser.add_argument("--strategy", default="normal", help="Prompt strategy (see prompt_strategies.json)")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temps.split(",")]

    # Load problems
    problems = []
    with open(args.problems) as f:
        for line in f:
            problems.append(json.loads(line))

    if args.filter_ids:
        filter_set = set(args.filter_ids.split(","))
        problems = [p for p in problems if p.get('task_id', '') in filter_set]

    print(f"Generating {args.samples} solutions/problem for {len(problems)} problems")
    print(f"Model: {args.model}, Strategy: {args.strategy}, Workers: {args.workers}")
    print(f"Temperatures: {temperatures}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    task_config = HumanEvalConfig()

    # Build all jobs: (problem, temperature) pairs
    jobs = []
    for problem in problems:
        for i in range(args.samples):
            temp = temperatures[i % len(temperatures)]
            jobs.append((problem, temp))

    print(f"Total requests: {len(jobs)}")
    start_time = time.time()

    total_pass = 0
    total_fail = 0
    total_error = 0
    per_problem = {}

    with open(args.output, 'a') as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for job_idx, (problem, temp) in enumerate(jobs):
                fut = executor.submit(
                    generate_one, client, args.model, problem,
                    task_config, temp, args.max_tokens, args.strategy,
                )
                futures[fut] = (problem['task_id'], job_idx)

            for i, fut in enumerate(as_completed(futures)):
                task_id, job_idx = futures[fut]
                record = fut.result()

                if record is None:
                    total_error += 1
                    continue

                fout.write(json.dumps(record) + '\n')
                if (i + 1) % 100 == 0:
                    fout.flush()

                if record['passed']:
                    total_pass += 1
                else:
                    total_fail += 1

                if task_id not in per_problem:
                    per_problem[task_id] = [0, 0]
                per_problem[task_id][0 if record['passed'] else 1] += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(jobs) - i - 1) / rate
                    print(f"  [{i+1}/{len(jobs)}] {rate:.1f} req/s, "
                          f"{total_pass}p/{total_fail}f/{total_error}err, "
                          f"~{remaining/60:.0f}min remaining")

        fout.flush()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. {total_pass} pass, {total_fail} fail, {total_error} errors")
    print(f"\nPer problem:")
    for tid in sorted(per_problem, key=lambda x: int(x.split('/')[1])):
        p, f = per_problem[tid]
        print(f"  {tid}: {p}p/{f}f")


if __name__ == "__main__":
    main()
