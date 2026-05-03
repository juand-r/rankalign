#!/usr/bin/env python3
"""
Generate solutions from API models across all prompt strategies.

Usage:
    .tools-venv/bin/python scripts/dataset_builder/generate_api_strategies.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from solution_generator.humaneval_config import HumanEvalConfig

EXCLUDED = {'HumanEval/53', 'HumanEval/145'}
SAMPLES = 10
STRATEGIES = ["beginner", "unusual", "refactorable", "different-style", "bad-style"]

API_MODELS = {
    'gpt-4o-mini': {'temps': [0.7, 1.0, 1.2], 'workers': 20},
    'gpt-5': {'temps': [1.0], 'workers': 10},
}


def generate_one(client, model, problem, task_config, temperature, max_tokens, strategy):
    """Generate and validate a single solution."""
    try:
        messages = task_config.make_prompt(problem, strategy=strategy)
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
            temperature=temperature, model=model, strategy=strategy,
        )
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    problems_path = "data/humaneval/problems.jsonl"
    output_path = "data/humaneval/solutions.jsonl"

    task_config = HumanEvalConfig()

    # Load problems
    problems = []
    with open(problems_path) as f:
        for line in f:
            p = json.loads(line)
            if p['task_id'] not in EXCLUDED:
                problems.append(p)
    print(f"Problems: {len(problems)}")

    # Set up OpenAI key
    sys.path.insert(0, 'packages/key_handler')
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
    client = OpenAI()

    total_generated = 0
    for model, cfg in API_MODELS.items():
        temps = cfg['temps']
        workers = cfg['workers']

        for strategy in STRATEGIES:
            jobs = []
            for p in problems:
                for i in range(SAMPLES):
                    temp = temps[i % len(temps)]
                    jobs.append((p, temp))

            print(f"\n{model} / {strategy}: {len(jobs)} requests ({workers} workers)")
            start = time.time()
            total_pass = 0
            total_fail = 0
            total_err = 0

            with open(output_path, 'a') as fout:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            generate_one, client, model, p,
                            task_config, t, 1024, strategy
                        ): p['task_id']
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
                            print(f"  [{i+1}/{len(jobs)}] {rate:.1f}/s, "
                                  f"{total_pass}p/{total_fail}f/{total_err}err, "
                                  f"~{remaining/60:.0f}m left")
                fout.flush()

            elapsed = time.time() - start
            batch_total = total_pass + total_fail
            total_generated += batch_total
            print(f"  Done: {total_pass}p/{total_fail}f/{total_err}err in {elapsed/60:.1f}m")

    print(f"\nAll done. {total_generated} total solutions generated.")


if __name__ == "__main__":
    main()
