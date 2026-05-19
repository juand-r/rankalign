"""Generate AST-mutated wrong solutions from the correct-solution mutation pool.

Applies rule-based AST mutations to correct solutions from the pool
(solutions.jsonl + solutions_spark_recleaned.jsonl + solutions_external.jsonl),
keeping only mutations that fail the HumanEval test suite with AssertionError
(i.e. plausible wrong solutions, not crashes or timeouts).

Output schema (data/humaneval/mutant_solutions_pool.jsonl):
  {task_id, source_model, source_strategy, source_solution, source_idx,
   mutator, mutator_description, mutated_solution, passed, error}

Resumable: skips (task_id, source_idx, mutator) already in output file.
Excludes intentional_bug strategy solutions (not in v2.1 correct distribution).

Usage:
    python scripts/dataset_builder/generate_mutant_solutions.py
"""

import ast
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data" / "humaneval"
POOL_SOURCES = [
    DATA_DIR / "solutions.jsonl",
    DATA_DIR / "solutions_external.jsonl",
]
SPARK_POOL = Path("/tmp/solutions_spark_recleaned.jsonl")
PROBLEMS_PATH = DATA_DIR / "problems.jsonl"
OUTPUT_PATH = DATA_DIR / "mutant_solutions_pool.jsonl"

EXCLUDE_STRATEGIES = {"intentional_bug"}


# ── Mutators ──────────────────────────────────────────────────────────────────

class CompareFlip(ast.NodeTransformer):
    name = "compare_flip"
    description = "Flip comparisons: > ↔ <, >= ↔ <=, == ↔ !="
    FLIP = {
        ast.Gt: ast.Lt, ast.Lt: ast.Gt,
        ast.GtE: ast.LtE, ast.LtE: ast.GtE,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        self.generic_visit(node)
        node.ops = [self.FLIP.get(type(op), type(op))() for op in node.ops]
        return node


class DenomPlusOne(ast.NodeTransformer):
    name = "denom_plus_one"
    description = "Division denominators +1: x / d → x / (d + 1)"

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            node.right = ast.BinOp(
                left=copy.deepcopy(node.right),
                op=ast.Add(),
                right=ast.Constant(value=1),
            )
        return node


class ReturnSlice(ast.NodeTransformer):
    name = "return_slice"
    description = "Truncate return values: return x → return x[:-1]"

    def visit_Return(self, node: ast.Return) -> ast.Return:
        self.generic_visit(node)
        if node.value is not None and not isinstance(node.value, ast.Constant):
            node.value = ast.Subscript(
                value=node.value,
                slice=ast.Slice(upper=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))),
                ctx=ast.Load(),
            )
        return node


class RangeMinusOne(ast.NodeTransformer):
    name = "range_minus_one"
    description = "range() upper bound -1: range(n) → range(n - 1)"

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            node.args[-1] = ast.BinOp(
                left=copy.deepcopy(node.args[-1]),
                op=ast.Sub(),
                right=ast.Constant(value=1),
            )
        return node


class AppendLoopVar(ast.NodeTransformer):
    name = "append_loop_var"
    description = "Append loop variable instead of accumulator in tracking loops"

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            return node
        loop_var = node.target.id
        for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "append"
                and len(stmt.value.args) == 1
                and isinstance(stmt.value.args[0], ast.Name)
                and stmt.value.args[0].id != loop_var
            ):
                stmt.value.args[0] = ast.Name(id=loop_var, ctx=ast.Load())
        return node


MUTATORS = [CompareFlip(), DenomPlusOne(), ReturnSlice(), RangeMinusOne(), AppendLoopVar()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_problems() -> dict:
    problems = {}
    with open(PROBLEMS_PATH) as f:
        for line in f:
            p = json.loads(line)
            problems[p["task_id"]] = p
    return problems


def load_pool(v2_1_tasks: set) -> list[dict]:
    """Load and deduplicate correct solutions for v2.1 tasks, excluding intentional_bug."""
    seen: set[tuple] = set()
    pool = []
    sources = list(POOL_SOURCES)
    if SPARK_POOL.exists():
        sources.append(SPARK_POOL)
    else:
        print(f"Warning: {SPARK_POOL} not found, skipping spark pool")

    for src in sources:
        with open(src) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("passed"):
                    continue
                if r["task_id"] not in v2_1_tasks:
                    continue
                if r.get("strategy") in EXCLUDE_STRATEGIES:
                    continue
                key = (r["task_id"], r["solution"][:80])
                if key in seen:
                    continue
                seen.add(key)
                pool.append(r)
    return pool


def apply_mutation(prompt: str, solution: str, mutator: ast.NodeTransformer) -> str | None:
    """Parse prompt+solution, apply mutator, return mutated body or None if no change."""
    src = prompt + "\n" + solution
    try:
        tree = ast.parse(src)
        original_src = ast.unparse(tree)
        mutated = mutator.visit(copy.deepcopy(tree))
        ast.fix_missing_locations(mutated)
        mutated_src = ast.unparse(mutated)
        if mutated_src == original_src:
            return None  # mutation was a no-op for this solution
        lines = mutated_src.split("\n")
        return "\n".join(lines[1:])  # strip the def line
    except Exception:
        return None


def run_tests(problem: dict, solution_body: str, timeout: int = 5) -> tuple[bool, str]:
    full_code = (
        problem["prompt"] + "\n" + solution_body
        + "\n\n" + problem["test"]
        + f"\ncheck({problem['entry_point']})"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = result.returncode == 0
        err = (result.stderr or result.stdout).strip().split("\n")[-1] if not passed else ""
        return passed, err
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, traceback.format_exc().strip().split("\n")[-1]
    finally:
        os.unlink(tmp_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    problems = load_problems()
    v2_1_tasks = set(problems.keys())  # all 164; we'll filter to those with pool solutions

    print("Loading mutation pool...", flush=True)
    pool = load_pool(v2_1_tasks)
    by_task: dict[str, list] = defaultdict(list)
    for r in pool:
        by_task[r["task_id"]].append(r)
    print(f"Pool: {len(pool)} solutions across {len(by_task)} tasks", flush=True)
    print(f"Excluded strategies: {EXCLUDE_STRATEGIES}", flush=True)

    # Load done set
    done: set[tuple] = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["task_id"], r["source_idx"], r["mutator"]))
        print(f"Resuming — {len(done)} already done", flush=True)

    total_candidates = sum(len(v) for v in by_task.values()) * len(MUTATORS)
    attempted = skipped_noop = skipped_pass = kept = 0

    with open(OUTPUT_PATH, "a") as fout:
        for task_id in sorted(by_task.keys()):
            problem = problems[task_id]
            solutions = by_task[task_id]
            task_kept = 0

            for source_idx, sol in enumerate(solutions):
                for mutator in MUTATORS:
                    key = (task_id, source_idx, mutator.name)
                    if key in done:
                        attempted += 1
                        continue

                    mutated_body = apply_mutation(problem["prompt"], sol["solution"], mutator)
                    attempted += 1

                    if mutated_body is None:
                        skipped_noop += 1
                        continue

                    passed, error = run_tests(problem, mutated_body)

                    if passed:
                        skipped_pass += 1
                        continue

                    # Only keep plausible failures (AssertionError)
                    if "AssertionError" not in error:
                        skipped_pass += 1
                        continue

                    row = {
                        "task_id": task_id,
                        "source_model": sol.get("model", ""),
                        "source_strategy": sol.get("strategy", ""),
                        "source_solution": sol["solution"],
                        "source_idx": source_idx,
                        "mutator": mutator.name,
                        "mutator_description": mutator.description,
                        "mutated_solution": mutated_body,
                        "passed": False,
                        "error": error,
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    kept += 1
                    task_kept += 1

            print(
                f"  {task_id}: {task_kept} mutants kept"
                f" | total kept={kept}, noop={skipped_noop}, pass={skipped_pass}"
                f" | {attempted}/{total_candidates} attempted",
                flush=True,
            )

    print(f"\nDone. {kept} plausible failing mutants → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
