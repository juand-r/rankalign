"""
Base classes for modular solution generation across tasks.

To add a new task, subclass TaskConfig and implement:
  - make_prompt(problem, strategy) -> str
  - validate(problem, raw_solution) -> (passed: bool, extracted_answer: str, error: str|None)
  - clean_solution(raw_solution) -> str  (optional, default is identity)

Then use generate_solutions() with your config.
"""

import json
import os
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


@dataclass
class TaskConfig(ABC):
    """Abstract base for task-specific generation config."""
    name: str  # e.g., "humaneval", "gsm8k"

    @abstractmethod
    def make_prompt(self, problem: dict, strategy: str = "normal") -> list[dict]:
        """Return OpenAI chat messages for generating a solution.

        Args:
            problem: dict with at least 'question' and any task-specific fields
            strategy: "normal" or "intentional_bug" (for generating wrong solutions)

        Returns: list of {"role": ..., "content": ...} messages
        """
        ...

    @abstractmethod
    def validate(self, problem: dict, raw_solution: str) -> tuple[bool, str, Optional[str]]:
        """Check if a solution is correct.

        Returns: (passed, extracted_answer, error_message_or_None)
        """
        ...

    def clean_solution(self, raw_solution: str) -> str:
        """Optional post-processing of raw LLM output. Default: strip."""
        return raw_solution.strip()

    def build_record(self, problem: dict, solution: str, raw_solution: str,
                     passed: bool, extracted_answer: str, error: Optional[str],
                     temperature: float, model: str, strategy: str) -> dict:
        """Build the output JSONL record. Override for task-specific fields."""
        return {
            'question_id': problem.get('task_id', problem.get('question_id', '')),
            'question': problem.get('question', problem.get('prompt', '')),
            'response': solution,
            'raw_response': raw_solution,
            'correct': passed,
            'extracted_answer': extracted_answer,
            'gold_answer': problem.get('gold_answer', ''),
            'error': error,
            'temperature': temperature,
            'model': model,
            'strategy': strategy,
            'task': self.name,
        }


@dataclass
class GenerationConfig:
    """Controls how solutions are generated."""
    model: str = "gpt-4o-mini"
    temperatures: list[float] = field(default_factory=lambda: [0.7, 1.0, 1.2])
    samples_per_problem: int = 20
    max_tokens: int = 2048
    strategy: str = "normal"  # "normal" or "intentional_bug"
    base_url: Optional[str] = None  # Custom API endpoint (e.g., vLLM server)
    api_key: Optional[str] = None  # Custom API key (e.g., for vLLM use "EMPTY")


def load_existing_counts(output_path: str) -> dict[str, dict[str, int]]:
    """Load existing solutions and count pass/fail per task_id."""
    counts = defaultdict(lambda: {'pass': 0, 'fail': 0})
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                # Support both field names: 'passed' (humaneval) and 'correct' (gsm8k)
                passed = r.get('passed', r.get('correct', False))
                qid = r.get('task_id', r.get('question_id', ''))
                key = 'pass' if passed else 'fail'
                counts[qid][key] += 1
    return counts


def generate_solutions(
    task: TaskConfig,
    problems: list[dict],
    gen_config: GenerationConfig,
    output_path: str,
    threshold: int = 10,
    target_side: str = "both",  # "both", "neg", "pos"
    resume: bool = True,
):
    """Generate solutions for a list of problems and append to output JSONL.

    Args:
        task: TaskConfig subclass with prompt/validation logic
        problems: list of dicts, each with 'question_id', 'question', etc.
        gen_config: generation parameters
        output_path: JSONL file to append to
        threshold: minimum count for qualification
        target_side: which side needs more solutions
        resume: skip problems that already have enough solutions
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    client_kwargs = {}
    if gen_config.base_url:
        client_kwargs['base_url'] = gen_config.base_url
    if gen_config.api_key:
        client_kwargs['api_key'] = gen_config.api_key
    client = OpenAI(**client_kwargs)

    existing = load_existing_counts(output_path) if resume else {}

    # Filter problems that need more solutions
    to_generate = []
    for p in problems:
        qid = p.get('task_id', p.get('question_id', ''))
        counts = existing.get(qid, {'pass': 0, 'fail': 0})
        if target_side == "neg" and counts['fail'] >= threshold:
            continue
        elif target_side == "pos" and counts['pass'] >= threshold:
            continue
        elif target_side == "both" and counts['pass'] >= threshold and counts['fail'] >= threshold:
            continue
        to_generate.append(p)

    print(f"[{task.name}] Generating {gen_config.samples_per_problem} solutions/problem "
          f"for {len(to_generate)}/{len(problems)} problems "
          f"(model={gen_config.model}, strategy={gen_config.strategy})")

    total_pass = 0
    total_fail = 0

    with open(output_path, 'a') as fout:
        for i, problem in enumerate(to_generate):
            qid = problem.get('task_id', problem.get('question_id', ''))
            n_pass = 0
            n_fail = 0

            for j in range(gen_config.samples_per_problem):
                temp = gen_config.temperatures[j % len(gen_config.temperatures)]
                try:
                    messages = task.make_prompt(problem, strategy=gen_config.strategy)
                    # GPT-5+ models require max_completion_tokens instead of max_tokens
                    token_param = ('max_completion_tokens'
                                   if gen_config.model.startswith('gpt-5')
                                   or gen_config.model.startswith('o')
                                   else 'max_tokens')
                    response = client.chat.completions.create(
                        model=gen_config.model,
                        messages=messages,
                        temperature=temp,
                        **{token_param: gen_config.max_tokens},
                    )
                    raw = response.choices[0].message.content.strip()
                    solution = task.clean_solution(raw)
                    passed, extracted_answer, error = task.validate(problem, solution)

                    record = task.build_record(
                        problem=problem,
                        solution=solution,
                        raw_solution=raw,
                        passed=passed,
                        extracted_answer=extracted_answer,
                        error=error,
                        temperature=temp,
                        model=gen_config.model,
                        strategy=gen_config.strategy,
                    )

                    fout.write(json.dumps(record) + '\n')
                    fout.flush()

                    if passed:
                        n_pass += 1
                    else:
                        n_fail += 1

                except Exception as e:
                    print(f"  Error on {qid} sample {j}: {e}")
                    traceback.print_exc()

            print(f"[{i+1}/{len(to_generate)}] {qid}: {n_pass} pass, {n_fail} fail")
            total_pass += n_pass
            total_fail += n_fail

    print(f"\nDone. Generated {total_pass + total_fail} solutions "
          f"({total_pass} pass, {total_fail} fail)")
    return total_pass, total_fail


def get_dataset_stats(output_path: str) -> dict:
    """Get comprehensive stats from a solutions JSONL file."""
    per_question = defaultdict(lambda: {
        'pass': 0, 'fail': 0, 'strategies': defaultdict(int), 'models': defaultdict(int)
    })
    total = 0

    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            qid = r.get('task_id', r.get('question_id', ''))
            total += 1
            passed = r.get('passed', r.get('correct', False))
            if passed:
                per_question[qid]['pass'] += 1
            else:
                per_question[qid]['fail'] += 1
            per_question[qid]['strategies'][r.get('strategy', 'unknown')] += 1
            per_question[qid]['models'][r.get('model', 'unknown')] += 1

    pass_counts = [v['pass'] for v in per_question.values()]
    fail_counts = [v['fail'] for v in per_question.values()]

    qualified = {qid: v for qid, v in per_question.items()
                 if v['pass'] >= 10 and v['fail'] >= 10}

    return {
        'total_solutions': total,
        'total_questions': len(per_question),
        'total_pass': sum(pass_counts),
        'total_fail': sum(fail_counts),
        'qualified_count': len(qualified),
        'per_question_pass': {
            'min': min(pass_counts) if pass_counts else 0,
            'median': sorted(pass_counts)[len(pass_counts)//2] if pass_counts else 0,
            'max': max(pass_counts) if pass_counts else 0,
        },
        'per_question_fail': {
            'min': min(fail_counts) if fail_counts else 0,
            'median': sorted(fail_counts)[len(fail_counts)//2] if fail_counts else 0,
            'max': max(fail_counts) if fail_counts else 0,
        },
        'per_question': dict(per_question),
    }


def print_stats(output_path: str):
    """Print a human-readable summary of dataset stats."""
    stats = get_dataset_stats(output_path)
    print(f"\n{'='*60}")
    print(f"Dataset: {output_path}")
    print(f"{'='*60}")
    print(f"Total solutions: {stats['total_solutions']}")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Correct: {stats['total_pass']} ({stats['total_pass']/stats['total_solutions']*100:.1f}%)")
    print(f"Incorrect: {stats['total_fail']} ({stats['total_fail']/stats['total_solutions']*100:.1f}%)")
    print(f"\nPer-question correct:   min={stats['per_question_pass']['min']}, "
          f"median={stats['per_question_pass']['median']}, max={stats['per_question_pass']['max']}")
    print(f"Per-question incorrect: min={stats['per_question_fail']['min']}, "
          f"median={stats['per_question_fail']['median']}, max={stats['per_question_fail']['max']}")
    print(f"\nQualified (>=10/10): {stats['qualified_count']}/{stats['total_questions']}")

    # Strategy breakdown
    all_strategies = defaultdict(lambda: {'pass': 0, 'fail': 0})
    for qid, q in stats['per_question'].items():
        for strat, count in q['strategies'].items():
            # Need to re-read to split by pass/fail per strategy
            pass
    print(f"{'='*60}")


def show_examples(output_path: str, n: int = 3, correct: bool = True):
    """Show n example solutions (correct or incorrect)."""
    examples = []
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            if r['correct'] == correct:
                examples.append(r)
            if len(examples) >= n:
                break

    label = "CORRECT" if correct else "INCORRECT"
    print(f"\n--- {n} {label} examples ---")
    for ex in examples:
        print(f"\nQ [{ex['question_id']}]: {ex['question'][:100]}...")
        print(f"Gold: {ex.get('gold_answer', 'N/A')}")
        print(f"Extracted: {ex.get('extracted_answer', 'N/A')}")
        print(f"Response (first 300 chars): {ex['response'][:300]}")
        print(f"Model: {ex.get('model')}, Strategy: {ex.get('strategy')}")
        print("---")
