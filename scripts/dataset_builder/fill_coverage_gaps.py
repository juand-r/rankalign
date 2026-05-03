#!/usr/bin/env python3
"""
Fill model coverage gaps: generate 10 samples from each model on every problem
where that model has <4 solutions. Ensures uniform model coverage across all problems.

Usage:
    # API models (run from laptop with OpenAI key):
    python scripts/dataset_builder/fill_coverage_gaps.py --mode api

    # vLLM models (run with spark vLLM forward on localhost:8000):
    python scripts/dataset_builder/fill_coverage_gaps.py --mode vllm --vllm-model MODEL_NAME

    # All vLLM models in sequence (cycling docker on spark):
    bash scripts/dataset_builder/run_coverage_cycle.sh
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from solution_generator.humaneval_config import HumanEvalConfig

EXCLUDED = {'HumanEval/53', 'HumanEval/145'}
MIN_COUNT = 4
SAMPLES = 10


def get_gaps(solutions_path, model_name):
    """Find problems where model has < MIN_COUNT solutions (excl intentional_bug)."""
    counts = defaultdict(int)
    with open(solutions_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('strategy') == 'intentional_bug':
                continue
            if r.get('model') == model_name:
                counts[r['task_id']] += 1

    all_problems = set(f'HumanEval/{i}' for i in range(164)) - EXCLUDED
    return sorted(
        [p for p in all_problems if counts[p] < MIN_COUNT],
        key=lambda x: int(x.split('/')[1])
    )


def generate_one(client, model, problem, task_config, temperature, max_tokens):
    """Generate and validate a single solution."""
    try:
        messages = task_config.make_prompt(problem, strategy="normal")
        # GPT-5+ requires max_completion_tokens
        token_param = ('max_completion_tokens'
                       if model.startswith('gpt-5') or model.startswith('o')
                       else 'max_tokens')
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **{token_param: max_tokens},
        )
        raw = response.choices[0].message.content.strip()
        solution = task_config.clean_solution(raw)
        passed, extracted_answer, error = task_config.validate(problem, solution)
        return task_config.build_record(
            problem=problem, solution=solution, raw_solution=raw,
            passed=passed, extracted_answer=extracted_answer, error=error,
            temperature=temperature, model=model, strategy="normal",
        )
    except Exception as e:
        return None


def run_model(client, model, problems_by_id, gap_ids, task_config, output_path,
              workers, temperatures, max_tokens):
    """Generate SAMPLES solutions for each gap problem."""
    problems = [problems_by_id[tid] for tid in gap_ids]
    jobs = []
    for p in problems:
        for i in range(SAMPLES):
            temp = temperatures[i % len(temperatures)]
            jobs.append((p, temp))

    print(f"  {model}: {len(gap_ids)} problems x {SAMPLES} = {len(jobs)} requests")
    start = time.time()
    total_pass = 0
    total_fail = 0
    total_err = 0

    with open(output_path, 'a') as fout:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate_one, client, model, p, task_config, t, max_tokens): p['task_id']
                for p, t in jobs
            }
            for i, fut in enumerate(as_completed(futures)):
                record = fut.result()
                if record is None:
                    total_err += 1
                    continue
                fout.write(json.dumps(record) + '\n')
                if (i + 1) % 100 == 0:
                    fout.flush()
                if record['passed']:
                    total_pass += 1
                else:
                    total_fail += 1

                if (i + 1) % 200 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    remaining = (len(jobs) - i - 1) / rate
                    print(f"    [{i+1}/{len(jobs)}] {rate:.1f}/s, {total_pass}p/{total_fail}f/{total_err}err, ~{remaining/60:.0f}m left")
        fout.flush()

    elapsed = time.time() - start
    print(f"  Done: {total_pass}p/{total_fail}f/{total_err}err in {elapsed/60:.1f}m")
    return total_pass, total_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "vllm"], required=True)
    parser.add_argument("--vllm-model", help="Model name for vLLM mode")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--problems", default="data/humaneval/problems.jsonl")
    parser.add_argument("--output", default="data/humaneval/solutions.jsonl")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    task_config = HumanEvalConfig()

    # Load problems
    problems_by_id = {}
    with open(args.problems) as f:
        for line in f:
            p = json.loads(line)
            problems_by_id[p['task_id']] = p

    if args.mode == "api":
        # Set up OpenAI key
        sys.path.insert(0, 'packages/key_handler')
        from key_handler import KeyHandler
        KeyHandler.set_env_key()
        client = OpenAI()

        api_models = {
            'gpt-4.1': {'temps': [0.7, 1.0, 1.2], 'workers': 20},
            'gpt-5': {'temps': [1.0], 'workers': 10},
            'gpt-5.5': {'temps': [1.0], 'workers': 10},
            'gpt-4o-mini': {'temps': [0.7, 1.0, 1.2], 'workers': 20},
        }

        for model, cfg in api_models.items():
            gaps = get_gaps(args.output, model)
            if not gaps:
                print(f"  {model}: no gaps, skipping")
                continue
            run_model(client, model, problems_by_id, gaps, task_config,
                      args.output, cfg['workers'], cfg['temps'], args.max_tokens)

    elif args.mode == "vllm":
        if not args.vllm_model:
            parser.error("--vllm-model required for vllm mode")
        client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")
        gaps = get_gaps(args.output, args.vllm_model)
        if not gaps:
            print(f"  {args.vllm_model}: no gaps, skipping")
            return
        run_model(client, args.vllm_model, problems_by_id, gaps, task_config,
                  args.output, args.workers, [0.7, 1.0, 1.2], args.max_tokens)


if __name__ == "__main__":
    main()
