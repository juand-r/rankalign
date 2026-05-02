"""
GSM8K task configuration for solution generation.

Validation: extract the final numeric answer from the model's response,
compare against the gold answer.
"""

import re
from .base import TaskConfig


# Standard GSM8K answer extraction (from openai/grade-school-math)
ANS_RE = re.compile(r"#### (\-?[\d\.\,]+)")
LAST_NUMBER_RE = re.compile(r"(-?[\d,]+\.?\d*)")


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

    def make_prompt(self, problem: dict, strategy: str = "normal") -> list[dict]:
        question = problem['question']

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
        else:
            return [
                {"role": "system", "content": (
                    "Solve the given math problem step by step. "
                    "Show your reasoning clearly, then end with 'The answer is <number>.'"
                )},
                {"role": "user", "content": question},
            ]

    def validate(self, problem: dict, raw_solution: str) -> tuple[bool, str, str | None]:
        gold = normalize_answer(problem['gold_answer'])
        extracted = extract_answer_from_response(raw_solution)
        extracted_norm = normalize_answer(extracted)

        if not extracted_norm:
            return False, extracted, "no_answer_found"

        passed = extracted_norm == gold
        return passed, extracted, None if passed else f"expected={gold}, got={extracted_norm}"
