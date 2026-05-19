"""Import external HumanEval solutions into solutions_external.jsonl.

Same sources as import_external_solutions.py, but:
  - Output goes to solutions_external.jsonl (NOT solutions.jsonl) so the
    main pool stays stable. See HUMANEVAL_BUILD_LOG.md for rationale.
  - jamesmurdza/humaneval-results: trusts the ✅/❌ labels in the markdown
    (no re-validation needed — labels are ground truth from the repo).
  - breath24/FailureBench: must validate (raw JSON has no pass/fail field).

Usage:
    python scripts/dataset_builder/import_external_solutions_mutation_pool.py \\
        --humaneval-results /tmp/humaneval-results \\
        --failurebench /tmp/FailureBench \\
        --problems data/humaneval/problems.jsonl \\
        --output data/humaneval/solutions_external.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def load_problems(path: str) -> dict:
    problems = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            problems[p["task_id"]] = p
    return problems


def validate_solution(prompt: str, solution: str, test_code: str, entry_point: str) -> tuple[bool, str | None]:
    full_code = prompt + solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        passed = result.returncode == 0
        error = result.stderr.strip() if not passed else None
        return passed, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def clean_solution_body(raw_code: str) -> str:
    """Extract function body and normalize indentation."""
    lines = raw_code.split("\n")
    body_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            for j in range(i, len(lines)):
                if lines[j].rstrip().endswith(":"):
                    body_start = j + 1
                    break
            break

    sol = "\n".join(lines[body_start:]) if body_start is not None and body_start < len(lines) else raw_code

    if sol.startswith("```"):
        sol_lines = sol.split("\n")[1:]
        if sol_lines and sol_lines[-1].strip() == "```":
            sol_lines = sol_lines[:-1]
        sol = "\n".join(sol_lines)

    result_lines = []
    for line in sol.split("\n"):
        if not line.strip():
            result_lines.append("")
        elif not line.startswith("    ") and line.strip():
            result_lines.append("    " + line)
        else:
            result_lines.append(line)
    return "\n".join(result_lines)


def parse_humaneval_results(repo_dir: str, problems: dict) -> list[dict]:
    """Parse jamesmurdza/humaneval-results using ✅/❌ labels — no re-validation."""
    model_dirs = {
        "codellama-34b-instruct": "CodeLlama-34b-Instruct",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4": "gpt-4",
    }
    run_pattern = re.compile(
        r"### ([✅❌]) Run (\d+).*?```python\n(.*?)```",
        re.DOTALL,
    )

    records = []
    for dir_name, model_name in model_dirs.items():
        model_dir = os.path.join(repo_dir, dir_name)
        if not os.path.exists(model_dir):
            print(f"  Skipping {model_name}: {model_dir} not found")
            continue

        n_pass = n_fail = 0
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith(".md"):
                continue
            problem_num = filename[:-3]
            task_id = f"HumanEval/{problem_num}"
            if task_id not in problems:
                continue
            problem = problems[task_id]

            with open(os.path.join(model_dir, filename)) as f:
                content = f.read()

            for match in run_pattern.finditer(content):
                passed = match.group(1) == "✅"
                raw_code = match.group(3).strip()
                solution = clean_solution_body(raw_code)
                if passed:
                    n_pass += 1
                else:
                    n_fail += 1
                records.append({
                    "task_id": task_id,
                    "prompt": problem["prompt"],
                    "solution": solution,
                    "raw_solution": raw_code,
                    "entry_point": problem["entry_point"],
                    "passed": passed,
                    "error": None,
                    "temperature": 0.2,
                    "model": model_name,
                    "strategy": "normal",
                    "source": "jamesmurdza/humaneval-results",
                })

        print(f"  {model_name}: {n_pass} pass, {n_fail} fail (labels from markdown)")

    return records


def parse_failurebench(repo_dir: str, problems: dict) -> list[dict]:
    """Parse breath24/FailureBench — must validate (no pass/fail labels in JSON)."""
    he_dir = os.path.join(repo_dir, "evaluation-results", "llm-generated-code", "HumanEval")
    if not os.path.exists(he_dir):
        print(f"  FailureBench HumanEval dir not found: {he_dir}")
        return []

    records = []
    for model_dir_name in sorted(os.listdir(he_dir)):
        raw_dir = os.path.join(he_dir, model_dir_name, "raw")
        if not os.path.isdir(raw_dir):
            continue
        model_name = model_dir_name.strip()
        n_pass = n_fail = 0

        files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
        print(f"  {model_name}: validating {len(files)} solutions...", flush=True)

        for filename in files:
            with open(os.path.join(raw_dir, filename)) as f:
                data = json.load(f)
            task_id = data["task_id"]
            if task_id not in problems:
                continue
            problem = problems[task_id]
            raw_code = data["llm_response"]
            solution = clean_solution_body(raw_code)
            passed, error = validate_solution(
                problem["prompt"], solution, problem["test"], problem["entry_point"]
            )
            if passed:
                n_pass += 1
            else:
                n_fail += 1
            records.append({
                "task_id": task_id,
                "prompt": problem["prompt"],
                "solution": solution,
                "raw_solution": raw_code,
                "entry_point": problem["entry_point"],
                "passed": passed,
                "error": error,
                "temperature": None,
                "model": model_name,
                "strategy": "normal",
                "source": "breath24/FailureBench",
            })

        print(f"  {model_name}: {n_pass} pass, {n_fail} fail")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humaneval-results", help="Path to jamesmurdza/humaneval-results clone")
    parser.add_argument("--failurebench", help="Path to breath24/FailureBench clone")
    parser.add_argument("--problems", required=True, help="Path to problems.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL (appends)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    print(f"Loaded {len(problems)} problems")

    all_records = []

    if args.humaneval_results:
        print(f"\nParsing humaneval-results (no re-validation)...")
        records = parse_humaneval_results(args.humaneval_results, problems)
        all_records.extend(records)
        print(f"  Subtotal: {len(records)}")

    if args.failurebench:
        print(f"\nParsing FailureBench (validation required — no labels in JSON)...")
        records = parse_failurebench(args.failurebench, problems)
        all_records.extend(records)
        print(f"  Subtotal: {len(records)}")

    pass_count = sum(1 for r in all_records if r["passed"])
    print(f"\nGrand total: {len(all_records)} ({pass_count} pass, {len(all_records)-pass_count} fail)")

    model_counts = Counter(r["model"] for r in all_records)
    print("\nPer model:")
    for m, c in model_counts.most_common():
        p = sum(1 for r in all_records if r["model"] == m and r["passed"])
        print(f"  {m}: {c} ({p} pass, {c-p} fail)")

    if not args.dry_run:
        out = Path(args.output)
        existing = sum(1 for _ in open(out)) if out.exists() else 0
        with open(out, "a") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        print(f"\nAppended {len(all_records)} records to {out} (was {existing} lines)")


if __name__ == "__main__":
    main()
