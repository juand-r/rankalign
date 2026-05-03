#!/usr/bin/env python3
"""
Unified solution generator for rankalign tasks.

Usage (from repo root):
    # Generate normal solutions for GSM8K train problems
    python scripts/dataset_builder/generate_solutions.py gsm8k \
        --problems data/gsm8k/gsm8k_train_problems.jsonl

    # Generate intentionally wrong solutions
    python scripts/dataset_builder/generate_solutions.py gsm8k \
        --problems data/gsm8k/gsm8k_train_problems.jsonl \
        --strategy intentional_bug --model gpt-4o

    # Show stats on existing solutions
    python scripts/dataset_builder/generate_solutions.py gsm8k \
        --stats --output data/gsm8k/solutions.jsonl

    # Show examples
    python scripts/dataset_builder/generate_solutions.py gsm8k \
        --examples --output data/gsm8k/solutions.jsonl

    # HumanEval
    python scripts/dataset_builder/generate_solutions.py humaneval \
        --problems data/humaneval/problems.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from solution_generator.base import (
    GenerationConfig, generate_solutions, print_stats, show_examples
)


TASK_CONFIGS = {
    'gsm8k': 'solution_generator.gsm8k_config:GSM8KConfig',
    'humaneval': 'solution_generator.humaneval_config:HumanEvalConfig',
}

DEFAULT_OUTPUTS = {
    'gsm8k': 'data/gsm8k/solutions.jsonl',
    'humaneval': 'data/humaneval/solutions.jsonl',
}


def load_task_config(task_name):
    module_path, class_name = TASK_CONFIGS[task_name].rsplit(':', 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def main():
    parser = argparse.ArgumentParser(description="Generate solutions for rankalign tasks")
    parser.add_argument("task", choices=list(TASK_CONFIGS.keys()),
                        help="Task to generate solutions for")
    parser.add_argument("--problems", type=str,
                        help="JSONL file with problems (each line has question_id, question, etc.)")
    parser.add_argument("--output", type=str, help="Output JSONL path (default: data/<task>/solutions.jsonl)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--strategy", type=str, default="normal",
                        choices=["normal", "intentional_bug"])
    parser.add_argument("--samples", type=int, default=20,
                        help="Solutions per problem")
    parser.add_argument("--temps", type=str, default="0.7,1.0,1.2",
                        help="Comma-separated temperature schedule")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Minimum pass/fail count for qualification")
    parser.add_argument("--target", type=str, default="both",
                        choices=["both", "neg", "pos"],
                        help="Which side needs more solutions")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--base-url", type=str, default=None,
                        help="Custom API base URL (e.g., http://localhost:8000/v1 for vLLM)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Custom API key (use 'EMPTY' for vLLM)")
    parser.add_argument("--filter-ids", type=str, default=None,
                        help="Comma-separated task IDs to generate for (e.g., 'HumanEval/0,HumanEval/6')")

    # Info modes (no generation)
    parser.add_argument("--stats", action="store_true", help="Show dataset stats and exit")
    parser.add_argument("--examples", action="store_true", help="Show example solutions and exit")
    parser.add_argument("--n-examples", type=int, default=3)

    args = parser.parse_args()

    output = args.output or DEFAULT_OUTPUTS.get(args.task, f'data/{args.task}/solutions.jsonl')

    if args.stats:
        print_stats(output)
        return

    if args.examples:
        show_examples(output, n=args.n_examples, correct=True)
        show_examples(output, n=args.n_examples, correct=False)
        return

    if not args.problems:
        parser.error("--problems is required for generation")

    # Load problems
    problems = []
    with open(args.problems) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"Loaded {len(problems)} problems from {args.problems}")

    # Filter to specific IDs if requested
    if args.filter_ids:
        filter_set = set(args.filter_ids.split(","))
        problems = [p for p in problems if p.get('task_id', p.get('question_id', '')) in filter_set]
        print(f"Filtered to {len(problems)} problems matching --filter-ids")

    task = load_task_config(args.task)
    gen_config = GenerationConfig(
        model=args.model,
        temperatures=[float(t) for t in args.temps.split(",")],
        samples_per_problem=args.samples,
        max_tokens=args.max_tokens,
        strategy=args.strategy,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    generate_solutions(
        task=task,
        problems=problems,
        gen_config=gen_config,
        output_path=output,
        threshold=args.threshold,
        target_side=args.target,
    )

    print_stats(output)


if __name__ == "__main__":
    main()
