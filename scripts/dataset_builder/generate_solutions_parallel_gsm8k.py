#!/usr/bin/env python3
"""
Parallel solution generator for GSM8K. Adapted from generate_solutions_parallel.py.

Two backends supported:
  - vLLM server: pass --base-url http://host:port/v1 (and optionally --api-key EMPTY)
  - OpenAI API: omit --base-url (uses OPENAI_API_KEY from env)

Usage:
    # vLLM (open weights on RunPod / Slurm GPU)
    python scripts/dataset_builder/generate_solutions_parallel_gsm8k.py \\
        --problems data/gsm8k/gsm8k_test_problems.jsonl \\
        --output data/gsm8k/v1/solutions.jsonl \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --base-url http://localhost:8000/v1 \\
        --strategy normal --samples 30 --temps 0.7,1.0,1.2 --workers 30

    # OpenAI API
    python scripts/dataset_builder/generate_solutions_parallel_gsm8k.py \\
        --problems data/gsm8k/gsm8k_test_problems.jsonl \\
        --output data/gsm8k/v1/solutions.jsonl \\
        --model gpt-4o-mini \\
        --strategy unusual --samples 30 --temps 0.7,1.0,1.2 --workers 10

Resume / skip-if-enough: if --skip-if-enough N is set, problems that already
have at least N pass AND N fail in the existing output JSONL are skipped.

Targeting: --target neg|pos|both restricts generation to problems still short
on a given side (used in gap-fill rounds).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from solution_generator.gsm8k_config import GSM8KConfig


def load_existing_counts(output_path):
    """Count pass/fail per task_id from existing JSONL (resume support).

    Counts only solutions for the same (model, strategy) by default — the
    parallel script is single-(model, strategy) per invocation, so this gives
    the right resume behavior. Set count_all_models=True to count across all.
    """
    counts = defaultdict(lambda: {'pass': 0, 'fail': 0})
    if not os.path.exists(output_path):
        return counts
    with open(output_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get('task_id') or r.get('question_id', '')
            passed = r.get('passed', r.get('correct', False))
            counts[qid]['pass' if passed else 'fail'] += 1
    return counts


def generate_one(client, model, problem, task_config, temperature, max_tokens, strategy="normal"):
    """Generate and validate a single solution. Returns a record dict or None on error."""
    try:
        messages = task_config.make_prompt(problem, strategy=strategy)
        # GPT-5+ models require max_completion_tokens; only temp=1.0 supported.
        token_param = ('max_completion_tokens'
                       if model.startswith('gpt-5') or model.startswith('o')
                       else 'max_tokens')
        kwargs = {
            'model': model,
            'messages': messages,
            'temperature': (1.0 if model.startswith('gpt-5') else temperature),
            token_param: max_tokens,
        }
        response = client.chat.completions.create(**kwargs)
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
        return {'__error__': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Parallel GSM8K solution generator (vLLM or OpenAI API)")
    parser.add_argument("--problems", required=True, help="problems.jsonl path")
    parser.add_argument("--output", required=True, help="Output JSONL (appends)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None,
                        help="vLLM server URL (e.g. http://localhost:8000/v1). Omit for OpenAI.")
    parser.add_argument("--api-key", default=None,
                        help="API key. Defaults: 'EMPTY' if --base-url set, else OPENAI_API_KEY env.")
    parser.add_argument("--samples", type=int, default=30, help="Samples per problem")
    parser.add_argument("--workers", type=int, default=30, help="Concurrent requests")
    parser.add_argument("--temps", default="0.7,1.0,1.2", help="Temperature schedule")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--filter-ids", help="Comma-separated question_ids (e.g. 'gsm8k_test_0,gsm8k_test_1')")
    parser.add_argument("--strategy", default="normal",
                        help="Prompt strategy (see solution_generator/gsm8k_strategies.json)")
    parser.add_argument("--skip-if-enough", type=int, default=None,
                        help="Skip problems already having >= N pass AND N fail in --output")
    parser.add_argument("--target", choices=['both', 'neg', 'pos'], default='both',
                        help="With --skip-if-enough: only generate for problems short on this side")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temps.split(",")]

    # Load problems
    problems = []
    with open(args.problems) as f:
        for line in f:
            problems.append(json.loads(line))

    if args.filter_ids:
        filter_set = set(args.filter_ids.split(","))
        problems = [p for p in problems if p.get('question_id', p.get('task_id', '')) in filter_set]

    # Resume / target filtering
    if args.skip_if_enough is not None:
        n = args.skip_if_enough
        existing = load_existing_counts(args.output)
        kept = []
        for p in problems:
            qid = p.get('question_id', p.get('task_id', ''))
            c = existing.get(qid, {'pass': 0, 'fail': 0})
            if args.target == 'both' and c['pass'] >= n and c['fail'] >= n:
                continue
            if args.target == 'neg' and c['fail'] >= n:
                continue
            if args.target == 'pos' and c['pass'] >= n:
                continue
            kept.append(p)
        print(f"Resume: skipped {len(problems) - len(kept)}/{len(problems)} problems "
              f"(target={args.target}, threshold={n})")
        problems = kept

    print(f"Generating {args.samples} solutions/problem for {len(problems)} problems")
    print(f"Model: {args.model}, Strategy: {args.strategy}, Workers: {args.workers}")
    print(f"Temperatures: {temperatures}")
    print(f"Backend: {'vLLM' if args.base_url else 'OpenAI'}  base_url={args.base_url or '(default)'}")

    client_kwargs = {}
    if args.base_url:
        client_kwargs['base_url'] = args.base_url
    api_key = args.api_key
    if api_key is None:
        api_key = "EMPTY" if args.base_url else os.environ.get("OPENAI_API_KEY")
    if api_key:
        client_kwargs['api_key'] = api_key
    client = OpenAI(**client_kwargs)

    task_config = GSM8KConfig()

    # Build all jobs: (problem, temperature) pairs
    jobs = []
    for problem in problems:
        for i in range(args.samples):
            temp = temperatures[i % len(temperatures)]
            jobs.append((problem, temp))

    print(f"Total requests: {len(jobs)}")
    if not jobs:
        print("Nothing to do.")
        return
    start_time = time.time()

    total_pass = 0
    total_fail = 0
    total_error = 0
    per_problem = {}

    with open(args.output, 'a') as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for job_idx, (problem, temp) in enumerate(jobs):
                qid = problem.get('question_id', problem.get('task_id', ''))
                fut = executor.submit(
                    generate_one, client, args.model, problem,
                    task_config, temp, args.max_tokens, args.strategy,
                )
                futures[fut] = (qid, job_idx)

            for i, fut in enumerate(as_completed(futures)):
                qid, job_idx = futures[fut]
                record = fut.result()

                if record is None or '__error__' in (record or {}):
                    total_error += 1
                    if (total_error <= 5) and record and '__error__' in record:
                        print(f"  [error] {qid}: {record['__error__'][:200]}")
                    continue

                fout.write(json.dumps(record) + '\n')
                if (i + 1) % 100 == 0:
                    fout.flush()

                if record['passed']:
                    total_pass += 1
                else:
                    total_fail += 1

                if qid not in per_problem:
                    per_problem[qid] = [0, 0]
                per_problem[qid][0 if record['passed'] else 1] += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(jobs) - i - 1) / rate if rate else 0
                    print(f"  [{i+1}/{len(jobs)}] {rate:.1f} req/s, "
                          f"{total_pass}p/{total_fail}f/{total_error}err, "
                          f"~{remaining/60:.0f}min remaining")

        fout.flush()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. {total_pass} pass, {total_fail} fail, {total_error} errors")
    if per_problem:
        n_qual = sum(1 for v in per_problem.values() if v[0] >= 10 and v[1] >= 10)
        print(f"Qualified (>=10p AND >=10f) in this run alone: {n_qual}/{len(per_problem)}")


if __name__ == "__main__":
    main()
