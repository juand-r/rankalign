"""
GSM8K task configuration for solution generation.

Validation: extract the final numeric answer from the model's response,
compare against the gold answer.

Multi-strategy prompting: when a strategy other than 'normal' or 'intentional_bug'
is requested, prompts are loaded from gsm8k_strategies.json (parallel to how
humaneval_config.py loads prompt_strategies.json).
"""

import json
import re
from pathlib import Path
from typing import Optional
from .base import TaskConfig


# Standard GSM8K answer extraction (from openai/grade-school-math)
ANS_RE = re.compile(r"#### (\-?[\d\.\,]+)")
LAST_NUMBER_RE = re.compile(r"(-?[\d,]+\.?\d*)")

_STRATEGIES_PATH = Path(__file__).parent / "gsm8k_strategies.json"


def _load_gsm8k_strategies():
    if not _STRATEGIES_PATH.exists():
        return {}
    with open(_STRATEGIES_PATH) as f:
        return json.load(f)


def extract_answer_from_response(response: str) -> str:
    """Extract the final numeric answer from a model response.

    Tries multiple patterns in order:
    1. #### <number> (GSM8K gold format)
    2. "The answer is <number>" (common CoT format)
    3. \\boxed{<number>} (LaTeX format)
    4. Last number in the response (fallback)
    """
    # Try #### format first
    match = ANS_RE.search(response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try "the answer is X" pattern
    answer_pattern = re.search(r"[Tt]he (?:final )?answer is[:\s]*\$?\\?boxed\{?(-?[\d,]+\.?\d*)\}?\$?", response)
    if answer_pattern:
        return answer_pattern.group(1).replace(",", "").strip()

    # Try \boxed{X}
    boxed = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", response)
    if boxed:
        return boxed.group(1).replace(",", "").strip()

    # Fallback: last number in the response
    numbers = LAST_NUMBER_RE.findall(response)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def normalize_answer(ans: str) -> str:
    """Normalize a numeric answer for comparison."""
    ans = ans.replace(",", "").replace("$", "").replace("%", "").strip()
    # Remove trailing .0 or .00
    if "." in ans:
        ans = ans.rstrip("0").rstrip(".")
    return ans


class GSM8KConfig(TaskConfig):
    """GSM8K math problem task configuration."""

    def __init__(self):
        super().__init__(name="gsm8k")
        self._strategies = _load_gsm8k_strategies()

    def make_prompt(self, problem: dict, strategy: str = "normal") -> list[dict]:
        question = problem['question']

        # Multi-strategy path: prompts loaded from gsm8k_strategies.json
        # (mirrors how humaneval_config.py loads prompt_strategies.json).
        if strategy in self._strategies:
            s = self._strategies[strategy]
            return [
                {"role": "system", "content": s["system_prompt"]},
                {"role": "user", "content": s["user_prompt"].format(question=question)},
            ]

        # Backward-compat: legacy intentional_bug branch (kept so older
        # code paths still work; v1 build excludes intentional_bug at filter
        # time, so it does not enter the v1 dataset).
        if strategy == "intentional_bug":
            return [
                {"role": "system", "content": (
                    "You are a math student who makes subtle calculation errors. "
                    "Solve the given problem step by step, but introduce a subtle arithmetic "
                    "or reasoning mistake somewhere in your work. The mistake should be plausible "
                    "— the kind a real student might make (wrong multiplication, forgetting a step, "
                    "off-by-one, misreading the problem). Show your work clearly. "
                    "End with 'The answer is <number>.' Do NOT mention that you made a mistake."
                )},
                {"role": "user", "content": question},
            ]

        # Fallback: same prompt as the legacy 'normal' branch (used when
        # gsm8k_strategies.json is missing). Keeps old call sites working.
        return [
            {"role": "system", "content": (
                "Solve the given math problem step by step. "
                "Show your reasoning clearly, then end with 'The answer is <number>.'"
            )},
            {"role": "user", "content": question},
        ]

    def clean_solution(self, raw_solution: str) -> str:
        """Light cleanup. GSM8K solutions need much less post-processing than
        code: no markdown fences to strip, no signature/body distinction. Just
        strip leading/trailing whitespace. The answer-extraction regex handles
        anything the model added after the answer line."""
        return raw_solution.strip()

    def build_record(self, problem: dict, solution: str, raw_solution: str,
                     passed: bool, extracted_answer: str, error: Optional[str],
                     temperature: float, model: str, strategy: str) -> dict:
        """Canonical schema (humaneval-style) for v1 pool rows."""
        return {
            'task_id': problem.get('question_id', problem.get('task_id', '')),
            'question': problem.get('question', ''),
            'solution': solution,
            'raw_solution': raw_solution,
            'extracted_answer': extracted_answer,
            'gold_answer': problem.get('gold_answer', ''),
            'passed': passed,
            'error': error,
            'temperature': temperature,
            'model': model,
            'strategy': strategy,
        }

    def validate(self, problem: dict, raw_solution: str) -> tuple[bool, str, "str | None"]:
        gold = normalize_answer(problem['gold_answer'])
        extracted = extract_answer_from_response(raw_solution)
        extracted_norm = normalize_answer(extracted)

        if not extracted_norm:
            return False, extracted, "no_answer_found"

        passed = extracted_norm == gold
        return passed, extracted, None if passed else f"expected={gold}, got={extracted_norm}"
