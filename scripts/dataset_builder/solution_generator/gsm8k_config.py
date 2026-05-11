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


# ====================================================================
# Dual-metric answer extraction — matches lm-evaluation-harness exactly:
#   https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
#   https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml
#
# We expose:
#   - strict_match(response) -> str | None  (uses LAST match of either gold-format `#### N`
#                                            or CoT-format `The answer is N.`)
#   - flexible_extract(response) -> str | None  (lm-eval-harness flexible filter,
#                                                last alternation match)
#
# `validate()` returns both strict_correct and flexible_correct flags so downstream
# can choose. `passed` (legacy boolean) is set to flexible_correct for back-compat
# with the older pool (which was generated with the buggy single-fallback extractor).
# ====================================================================

# gsm8k.yaml (strict-match)
_RE_STRICT_HASH = re.compile(r"#### (\-?[0-9\.\,]+)")
# gsm8k-cot-zeroshot.yaml (strict-match) — literal trailing period required
_RE_STRICT_ANSWER_IS = re.compile(r"The answer is (\-?[0-9\.\,]+)\.")
# gsm8k.yaml flexible-extract — group_select=-1 (last match)
_RE_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

# Legacy names kept so any old callers don't break (used to be the simple fallback)
ANS_RE = _RE_STRICT_HASH
LAST_NUMBER_RE = re.compile(r"(-?[\d,]+\.?\d*)")

_STRATEGIES_PATH = Path(__file__).parent / "gsm8k_strategies.json"


def _load_gsm8k_strategies():
    if not _STRATEGIES_PATH.exists():
        return {}
    with open(_STRATEGIES_PATH) as f:
        return json.load(f)


def _last_match(regex, text: str) -> Optional[str]:
    """Return last match's first non-empty group, or None."""
    matches = list(regex.finditer(text))
    if not matches:
        return None
    for g in matches[-1].groups():
        if g:
            return g
    return matches[-1].group(0)


def strict_match(response: str) -> Optional[str]:
    """Strict-match extraction. Returns captured number string or None.

    Accepts EITHER `#### N` OR `The answer is N.` (last occurrence; prefers
    explicit "The answer is" form when both are present)."""
    ans = _last_match(_RE_STRICT_ANSWER_IS, response)
    if ans is not None:
        return ans
    return _last_match(_RE_STRICT_HASH, response)


def flexible_extract(response: str) -> Optional[str]:
    """Flexible extraction: lm-eval-harness flexible filter (last alternation match)."""
    return _last_match(_RE_FLEXIBLE, response)


def extract_answer_from_response(response: str) -> str:
    """Legacy single-string extractor (now delegates to strict, then flexible).

    Returns the strict match if present, otherwise the flexible match, otherwise "".
    Callers should prefer the dual-metric API (`strict_match` + `flexible_extract`)
    so they can store both labels per row.
    """
    ans = strict_match(response)
    if ans is not None:
        return ans.replace(",", "").strip()
    ans = flexible_extract(response)
    if ans is not None:
        return ans.replace(",", "").strip()
    return ""


def _legacy_unused_old_extractor_kept_for_history(response: str) -> str:
    """Original buggy extractor (kept for historical reference / debugging only)."""
    match = ANS_RE.search(response)
    if match:
        return match.group(1).replace(",", "").strip()
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
        """Canonical schema (humaneval-style) for v1 pool rows.

        v2 additions (back-compat): we also store strict_extracted, flexible_extracted,
        strict_correct, flexible_correct so downstream consumers can use either
        lm-eval-harness metric. `passed` (legacy) = flexible_correct, matching the
        v1 pool's labeling convention.
        """
        # Compute dual metrics
        strict = strict_match(raw_solution)
        flex = flexible_extract(raw_solution)
        gold = normalize_answer(problem.get('gold_answer', ''))
        strict_norm = normalize_answer(strict) if strict else ""
        flex_norm = normalize_answer(flex) if flex else ""
        strict_correct = (strict_norm != "") and (strict_norm == gold)
        flexible_correct = (flex_norm != "") and (flex_norm == gold)

        return {
            'task_id': problem.get('question_id', problem.get('task_id', '')),
            'question': problem.get('question', ''),
            'solution': solution,
            'raw_solution': raw_solution,
            'extracted_answer': extracted_answer,
            'gold_answer': problem.get('gold_answer', ''),
            'passed': passed,                      # legacy = flexible_correct (back-compat)
            'strict_extracted': strict,
            'flexible_extracted': flex,
            'strict_correct': strict_correct,
            'flexible_correct': flexible_correct,
            'error': error,
            'temperature': temperature,
            'model': model,
            'strategy': strategy,
        }

    def validate(self, problem: dict, raw_solution: str) -> tuple[bool, str, "str | None"]:
        """Return (passed, extracted_str, error).

        passed = flexible_correct (matches lm-eval-harness flexible-extract metric,
        and matches the v1 pool's `correct` convention for back-compat).
        Both strict and flexible per-row labels land in the record via build_record.
        """
        gold = normalize_answer(problem.get('gold_answer', ''))
        flex = flexible_extract(raw_solution)
        flex_norm = normalize_answer(flex) if flex else ""

        if flex_norm == "":
            return False, "", "no_answer_found"

        passed = flex_norm == gold
        return passed, flex or "", None if passed else f"expected={gold}, got={flex_norm}"
