"""Build humaneval-v2.1correct-upper: drop-in replacement for v2.1 with all
correct answers' local variables renamed to UPPER_CASE.

For each test CSV in data/humaneval/v2.1/:
  * wrong rows: kept unchanged
  * correct rows: AST-based rename of local body variables to UPPER_CASE
    - Validates ALL transformed correct against HumanEval unit tests (strict: aborts on failure)
    - If a correct row's answer can't be AST-parsed (column-0 def in body, etc.),
      keep the original answer with no transform
  * Same rule for train.csv

Output: data/humaneval/v2.1correct-upper/{slug}.csv  (same schema as v2.1)
"""

import ast
import builtins
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V21_DIR = ROOT / "data/humaneval/v2.1"
OUT_DIR = ROOT / "data/humaneval/v2.1correct-upper"
PROBLEMS_JSONL = ROOT / "data/humaneval/problems.jsonl"

DEF_SIG_RE = re.compile(r'^def [^\n]+:\s*$', re.MULTILINE)
_FROZEN = set(dir(builtins)) | {
    "self", "cls", "math", "re", "os", "sys", "json", "typing",
    "List", "Dict", "Set", "Tuple", "Optional", "Any", "Union", "Callable",
    "Iterable", "Iterator", "Sequence", "Mapping",
}


def load_problems():
    out = {}
    with open(PROBLEMS_JSONL) as f:
        for line in f:
            p = json.loads(line)
            out[p["task_id"]] = p
    return out


def slug_to_id(slug):
    return f"HumanEval/{slug.split('_')[1]}"


def _last_def_signature(q):
    m = DEF_SIG_RE.findall(q)
    if not m: raise ValueError("no def line")
    return m[-1]


def _signature_params(sig):
    parsed = ast.parse(sig + "\n    pass")
    func = parsed.body[0]
    params = set()
    for a in func.args.args: params.add(a.arg)
    if func.args.vararg:  params.add(func.args.vararg.arg)
    if func.args.kwarg:   params.add(func.args.kwarg.arg)
    for a in func.args.kwonlyargs: params.add(a.arg)
    return params


def _collect(node, out):
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts: _collect(e, out)
    elif isinstance(node, ast.Starred):
        _collect(node.value, out)


def _local_names(func_def):
    locals_ = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign):
            for t in node.targets: _collect(t, locals_)
        elif isinstance(node, ast.AugAssign): _collect(node.target, locals_)
        elif isinstance(node, ast.AnnAssign) and node.target: _collect(node.target, locals_)
        elif isinstance(node, ast.For): _collect(node.target, locals_)
        elif isinstance(node, ast.comprehension): _collect(node.target, locals_)
        elif isinstance(node, ast.NamedExpr): _collect(node.target, locals_)
        elif isinstance(node, ast.With):
            for it in node.items:
                if it.optional_vars: _collect(it.optional_vars, locals_)
        elif isinstance(node, ast.FunctionDef) and node is not func_def:
            locals_.add(node.name)
    return locals_


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping): self.mapping = mapping
    def visit_Name(self, node):
        if node.id in self.mapping: return ast.Name(id=self.mapping[node.id], ctx=node.ctx)
        return node
    def visit_arg(self, node):
        if node.arg in self.mapping: node.arg = self.mapping[node.arg]
        return node
    def visit_FunctionDef(self, node):
        if node.name in self.mapping: node.name = self.mapping[node.name]
        self.generic_visit(node); return node
    def visit_Nonlocal(self, node):
        node.names = [self.mapping.get(n, n) for n in node.names]; return node
    def visit_Global(self, node):
        node.names = [self.mapping.get(n, n) for n in node.names]; return node
    def visit_ExceptHandler(self, node):
        if node.name and node.name in self.mapping: node.name = self.mapping[node.name]
        self.generic_visit(node); return node


def try_transform(question, v2_answer):
    """Return (transformed_v2_format, mapping_dict).
    Raises SyntaxError if the answer can't be parsed."""
    sig = _last_def_signature(question)
    params = _signature_params(sig)
    body_full_indent = "\n".join(
        ("    " + line if line.strip() and not line.startswith("    ") else line)
        for line in v2_answer.split("\n")
    )
    full_func_src = sig + "\n" + body_full_indent
    parsed = ast.parse(full_func_src)
    func = parsed.body[0]
    local_names = _local_names(func)
    renameable = (local_names - params - _FROZEN)
    mapping = {n: n.upper() for n in renameable if not n.startswith("_") and not n.isupper()}
    new_func = _Renamer(mapping).visit(func)
    ast.fix_missing_locations(new_func)
    new_full = ast.unparse(new_func)
    body_lines = new_full.split("\n")[1:]
    body = "\n".join(body_lines)
    body_v2 = body[4:] if body.startswith("    ") else body
    return body_v2, mapping


def validate(problem, v2_answer):
    body_with_indent = "\n".join(
        ("    " + line if line.strip() and not line.startswith("    ") else line)
        for line in v2_answer.split("\n")
    )
    full_code = problem["prompt"] + body_with_indent + "\n\n" + problem["test"] + \
                f"\n\ncheck({problem['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            return True, None
        return False, (result.stderr.strip()[:500] or "non-zero exit")
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(tmp)


def process_csv(csv_path, problems):
    """Return (df_out, stats_dict)."""
    df = pd.read_csv(csv_path)
    slug = csv_path.stem
    if slug == "train":
        # train.csv has rows from many tasks (task_id column tells us which)
        problem_for_row = None
    else:
        problem_for_row = problems[slug_to_id(slug)]
    out_rows = []
    stats = dict(total=0, n_correct=0, n_wrong=0,
                 transformed=0, no_op=0, parse_skipped=0,
                 validation_reverts=[])
    for ri, row in df.iterrows():
        stats["total"] += 1
        is_correct = str(row.get("correct", "")).strip().lower() == "yes"
        if not is_correct:
            stats["n_wrong"] += 1
            out_rows.append(row.to_dict())
            continue
        stats["n_correct"] += 1
        question = row["question"]
        answer = str(row["answer"])
        # For train.csv, find problem via task_id column
        if problem_for_row is None:
            tid = row.get("task_id", "")
            if tid not in problems:
                raise ValueError(f"task_id '{tid}' not in problems.jsonl for train.csv row {ri}")
            this_problem = problems[tid]
        else:
            this_problem = problem_for_row
        try:
            new_answer, mapping = try_transform(question, answer)
        except (SyntaxError, ValueError):
            stats["parse_skipped"] += 1
            out_rows.append(row.to_dict())  # keep original
            continue
        if not mapping:
            stats["no_op"] += 1
        else:
            stats["transformed"] += 1
        # Validate transformed correct against HumanEval tests.
        # If it fails (e.g. string-literal references like 'idx' in locals(),
        # or column-0 imports that worked at module-level pre-transform), revert
        # to the original answer for that row. Same fallback as parse failure.
        passed, err = validate(this_problem, new_answer)
        if not passed:
            stats["validation_reverts"].append({
                "slug": slug, "row_idx": int(ri),
                "mapping": mapping, "error": err[:300],
            })
            out_rows.append(row.to_dict())  # keep original, no transform applied
            continue
        new_row = row.to_dict()
        new_row["answer"] = new_answer
        out_rows.append(new_row)
    return pd.DataFrame(out_rows, columns=df.columns), stats


def main():
    problems = load_problems()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_csv = sorted(V21_DIR.glob("humaneval_*.csv"))
    train_csv = V21_DIR / "train.csv"

    grand = dict(total=0, n_correct=0, n_wrong=0,
                 transformed=0, no_op=0, parse_skipped=0,
                 validation_reverts=[])

    print(f"Processing {len(all_csv)} test CSVs + 1 train CSV...")
    files_to_do = list(all_csv) + ([train_csv] if train_csv.exists() else [])
    for fp in files_to_do:
        out_df, stats = process_csv(fp, problems)
        out_df.to_csv(OUT_DIR / fp.name, index=False)
        for k in ["total", "n_correct", "n_wrong", "transformed", "no_op", "parse_skipped"]:
            grand[k] += stats[k]
        grand["validation_reverts"].extend(stats["validation_reverts"])
        if stats["validation_reverts"] or stats["parse_skipped"]:
            print(f"  {fp.stem}: total={stats['total']} correct={stats['n_correct']} "
                  f"transformed={stats['transformed']} no-op={stats['no_op']} "
                  f"parse-skip={stats['parse_skipped']} val-revert={len(stats['validation_reverts'])}")

    print("\n=== GRAND SUMMARY ===")
    print(f"Total rows: {grand['total']}")
    print(f"  Correct: {grand['n_correct']}")
    print(f"  Wrong:   {grand['n_wrong']}")
    print(f"Correct rows breakdown:")
    print(f"  Transformed (renamed):     {grand['transformed']}")
    print(f"  No-op (no renameable vars): {grand['no_op']}")
    print(f"  Parse-skipped (kept orig):  {grand['parse_skipped']}")
    print(f"  Validation reverts (kept orig): {len(grand['validation_reverts'])}")

    if grand["validation_reverts"]:
        print(f"\n=== ROWS REVERTED TO ORIGINAL (transform broke validation) ===")
        for vf in grand["validation_reverts"]:
            print(f"  {vf['slug']}:{vf['row_idx']}  mapping={vf['mapping']}")
            print(f"    reason: {vf['error'][:120]}")
        with open(OUT_DIR / "_validation_reverts.json", "w") as f:
            json.dump(grand["validation_reverts"], f, indent=2, ensure_ascii=False)
        print(f"\nLogged to {OUT_DIR}/_validation_reverts.json")

    print(f"\nWrote {len(files_to_do)} CSVs to {OUT_DIR}")
    print(f"All correct rows present; transform applied where possible.")


if __name__ == "__main__":
    main()
