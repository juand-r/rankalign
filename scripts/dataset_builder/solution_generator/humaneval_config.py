"""
HumanEval task configuration for solution generation.

Validation: run the generated code against HumanEval unit tests.

Prompt strategies are defined in prompt_strategies.json (same directory).
Each strategy maps to a system_prompt and user_prompt template.
"""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from .base import TaskConfig

_STRATEGIES_PATH = Path(__file__).parent / "prompt_strategies.json"


def _load_strategies():
    with open(_STRATEGIES_PATH) as f:
        return json.load(f)


class HumanEvalConfig(TaskConfig):
    """HumanEval code correctness task configuration."""

    def __init__(self):
        super().__init__(name="humaneval")
        self._strategies = _load_strategies()

    def make_prompt(self, problem: dict, strategy: str = "normal") -> list[dict]:
        prompt = problem.get('prompt', problem.get('question', ''))

        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(self._strategies.keys())}. "
                f"See {_STRATEGIES_PATH}"
            )

        s = self._strategies[strategy]
        return [
            {"role": "system", "content": s["system_prompt"]},
            {"role": "user", "content": s["user_prompt"].format(prompt=prompt)},
        ]

    def clean_solution(self, raw_solution: str) -> str:
        """Clean up raw LLM output to extract just the function body."""
        sol = raw_solution

        # Strip markdown code blocks
        if sol.startswith("```"):
            lines = sol.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            sol = "\n".join(lines)

        # Strip trailing ``` and any explanation text after it
        lines = sol.split("\n")
        cut_at = None
        for i, line in enumerate(lines):
            if line.strip() == "```":
                cut_at = i
                break
        if cut_at is not None:
            lines = lines[:cut_at]
            sol = "\n".join(lines)

        # If solution contains full function signature, extract body only
        lines = sol.split("\n")
        body_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                for j in range(i, len(lines)):
                    if lines[j].rstrip().endswith(":"):
                        body_start = j + 1
                        break
                break

        if body_start is not None and body_start < len(lines):
            sol = "\n".join(lines[body_start:])

        # Ensure proper indentation (4 spaces for function body)
        lines = sol.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return sol

        first_nonempty = non_empty[0]
        needs_indent = not first_nonempty.startswith("    ")

        if needs_indent and len(non_empty) >= 2:
            # Detect pattern: is the code consistently at base indent 0?
            # Pattern A: "result = []\nfor i in range(n):\n    if ..." — all top-level at 0
            # Pattern B: "numbers.sort()\n    for i in range(n):" — first at 0, rest at 4+
            second_indent = len(non_empty[1]) - len(non_empty[1].lstrip())
            if second_indent == 0:
                # Pattern A: code is structured from indent 0, shift everything by 4
                result_lines = []
                for line in lines:
                    if line.strip() == "":
                        result_lines.append("")
                    else:
                        result_lines.append("    " + line)
                return "\n".join(result_lines)

        # Default: add 4 spaces only to lines that aren't already indented
        if needs_indent:
            result_lines = []
            for line in lines:
                if line.strip() == "":
                    result_lines.append("")
                elif not line.startswith("    ") and line.strip():
                    result_lines.append("    " + line)
                else:
                    result_lines.append(line)
        else:
            result_lines = lines

        return "\n".join(result_lines)

    def build_record(self, problem: dict, solution: str, raw_solution: str,
                     passed: bool, extracted_answer: str, error: Optional[str],
                     temperature: float, model: str, strategy: str) -> dict:
        """Match existing solutions.jsonl format (task_id/passed, not question_id/correct)."""
        return {
            'task_id': problem.get('task_id', problem.get('question_id', '')),
            'prompt': problem.get('prompt', problem.get('question', '')),
            'solution': solution,
            'raw_solution': raw_solution,
            'entry_point': problem.get('entry_point', ''),
            'passed': passed,
            'error': error,
            'temperature': temperature,
            'model': model,
            'strategy': strategy,
        }

    def validate(self, problem: dict, raw_solution: str) -> tuple[bool, str, str | None]:
        """Run solution against HumanEval unit tests."""
        prompt = problem.get('prompt', problem.get('question', ''))
        test_code = problem['test']
        entry_point = problem['entry_point']

        full_code = prompt + raw_solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            passed = result.returncode == 0
            error = result.stderr.strip() if not passed else None
            return passed, "pass" if passed else "fail", error
        except subprocess.TimeoutExpired:
            return False, "timeout", "TIMEOUT"
        except Exception as e:
            return False, "error", str(e)
        finally:
            os.unlink(tmp_path)
