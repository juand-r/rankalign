"""correct_multi — libcst, comment-preserving stylization of correct HumanEval
answers for humaneval-v2.1correct-multi.

Correctness model (see BUILD_PLAN.md / red_team_brief.md):
  * The rename SET and the re-indent are NOT reimplemented — `_local_names`,
    `_signature_params`, `_FROZEN`, `_last_def_signature` are imported verbatim
    from the trusted ast reference `build_humaneval_v2_1_correct_upper.py`
    (DRY). ast decides *which* identifiers are renameable.
  * libcst is the rewrite/serialization engine ONLY (it preserves comments &
    formatting; ast.unparse does not). It is not trusted for scope.
  * Parse-source identity (N3): ast scope and libcst both parse the *identical*
    string `sig + "\n" + body_full_indent`.
  * Occurrence-level verification (N3): the multiset of identifier tokens in
    the libcst output must equal that of the ast-reference renamer's output —
    proves the rename hit exactly the same occurrences (robust to shadowing).
  * Every Axis-2 transform is semantics-preserving *by construction*; the
    HumanEval unit-test re-validation (imported `validate`) is the backstop.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider

# --- DRY: import the trusted ast reference (no reimplementation) -------------
_RANKALIGN_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_RANKALIGN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_RANKALIGN_SCRIPTS))
from dataset_builder.build_humaneval_v2_1_correct_upper import (  # noqa: E402
    _FROZEN,
    _last_def_signature,
    _local_names,
    _signature_params,
    _Renamer as _AstRenamer,
)

import keyword  # noqa: E402

RENAME_SCHEMES = ("upper", "camel", "verbose", "cryptic", "hungarian", "numbered")


# --- shared front-end (verbatim parity with the reference) ------------------

def build_full_func_src(question: str, v2_answer: str) -> tuple[str, set, str]:
    """Replicate build_humaneval_v2_1_correct_upper.try_transform lines 120-128
    exactly. Returns (full_func_src, params, body_full_indent).

    full_func_src is THE string both ast scope and libcst parse (N3).
    """
    sig = _last_def_signature(question)
    params = _signature_params(sig)
    body_full_indent = "\n".join(
        ("    " + line if line.strip() and not line.startswith("    ") else line)
        for line in v2_answer.split("\n")
    )
    full_func_src = sig + "\n" + body_full_indent
    return full_func_src, params, body_full_indent


def _rename_one(name: str, scheme: str, idx: int) -> str:
    """Deterministic per (scheme, name, idx-within-sorted-renameable).

    idx makes `cryptic` collision-free without type inference. These flavors
    are approximations; the canary measures which actually lower log P(y|x).
    """
    if scheme == "upper":
        return name.upper()
    if scheme == "camel":
        parts = name.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    if scheme == "verbose":
        return f"the_{name}_value"
    if scheme == "cryptic":
        return f"v{idx}"
    if scheme == "hungarian":
        return f"x_{name}"
    if scheme == "numbered":
        return f"{name}{idx}"
    raise ValueError(f"unknown scheme {scheme!r}")


def compute_rename_map(full_func_src: str, params: set, scheme: str) -> dict[str, str]:
    """ast-derived rename map (mirrors reference lines 129-131, generalized to
    the 6 schemes). Guarantees the target set is disjoint from params, _FROZEN,
    keywords, original locals, and itself (injective) — collisions dropped
    conservatively. Reuses `_local_names` verbatim (the trusted scope walk)."""
    parsed = ast.parse(full_func_src)
    func = parsed.body[0]
    local_names = _local_names(func)
    renameable = sorted(local_names - params - _FROZEN)
    forbidden = set(params) | set(_FROZEN) | set(local_names) | set(keyword.kwlist)
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()
    for idx, n in enumerate(renameable):
        if n.startswith("_") or n.isupper():
            continue  # reference guard (line 131): skip dunder/already-UPPER
        tgt = _rename_one(n, scheme, idx)
        if (
            tgt == n
            or not tgt.isidentifier()
            or keyword.iskeyword(tgt)
            or tgt in forbidden
            or tgt in used_targets
        ):
            continue  # conservative: skip rather than risk a collision
        mapping[n] = tgt
        used_targets.add(tgt)
    return mapping


# --- libcst renamer: mirrors the reference ast _Renamer semantics -----------
# ast.Name == a variable reference/binding (never an attribute label or a
# call-site keyword). libcst Name also appears as Attribute.attr and
# Arg.keyword — those must NOT be renamed (the reference never did, since
# ast.Attribute.attr / ast.keyword.arg are plain strings, not ast.Name).

class _LibcstRenamer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def leave_Name(self, orig: cst.Name, updated: cst.Name) -> cst.BaseExpression:
        if orig.value not in self.mapping:
            return updated
        parent = self.get_metadata(ParentNodeProvider, orig)
        # skip attribute label `x.attr` and call keyword `f(kw=...)`
        if isinstance(parent, cst.Attribute) and parent.attr is orig:
            return updated
        if isinstance(parent, cst.Arg) and parent.keyword is orig:
            return updated
        return updated.with_changes(value=self.mapping[orig.value])


def apply_rename(full_func_src: str, mapping: dict[str, str]) -> str:
    """libcst rename on the IDENTICAL source ast scoped (N3)."""
    if not mapping:
        return full_func_src
    wrapper = MetadataWrapper(cst.parse_module(full_func_src))
    return wrapper.visit(_LibcstRenamer(mapping)).code


def _name_multiset(src: str) -> Counter:
    """Multiset of ast.Name ids + def/arg names — identifier occurrences."""
    tree = ast.parse(src)
    c: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            c[node.id] += 1
        elif isinstance(node, ast.arg):
            c[node.arg] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            c[node.name] += 1
    return c


def verify_occurrence_equivalence(full_func_src: str, mapping: dict[str, str],
                                  libcst_out: str) -> bool:
    """N3: libcst output must rename the SAME occurrences as the trusted ast
    renamer. Compare identifier multisets (robust to formatting/comments and
    to shadowing — a positional, not name-set, check)."""
    parsed = ast.parse(full_func_src)
    func = parsed.body[0]
    ast_renamed = _AstRenamer(mapping).visit(func)
    ast.fix_missing_locations(ast_renamed)
    ast_ref_src = ast.unparse(ast_renamed)
    return _name_multiset(ast_ref_src) == _name_multiset(libcst_out)


# --- v2-format store (no ast.unparse; replicate reference contract) ---------

def to_v2_format(rewritten_full_func_src: str) -> str:
    """Drop the def line; restore v2's line-1-at-column-0 (reference lines
    135-137: `body[4:] if body.startswith('    ')`). Comments preserved."""
    body = "\n".join(rewritten_full_func_src.split("\n")[1:])
    return body[4:] if body.startswith("    ") else body


# --- Axis-2 composable transforms (each safe by construction) ---------------

def _reserved_temp(post_rename_src: str, k: int = 0) -> str:
    """A `_`-prefixed name guaranteed unused & never a rename target (the
    reference map skips `_`-prefixed). Checked vs the POST-rename universe (N2)."""
    present = set(_name_multiset(post_rename_src))
    while f"_T{k}" in present:
        k += 1
    return f"_T{k}"


class _OutermostFuncStmt(cst.CSTTransformer):
    """Mixin: only act on statements in the OUTERMOST FunctionDef's own scope
    (FunctionDef-depth == 1), never inside a nested helper `def` (depth ≥ 2).
    Returns inside if/for within the solution stay depth==1 (those ARE the
    solution's). Without this, post-order leave_* mutates a nested helper's
    return → wrong code region → confounds per-transform ΔlogP attribution."""
    def __init__(self):
        self.done = False
        self._fdepth = 0

    def visit_FunctionDef(self, node):  # noqa: N802
        self._fdepth += 1

    def leave_FunctionDef(self, orig, updated):  # noqa: N802
        self._fdepth -= 1
        return updated

    def _outermost(self) -> bool:
        return self._fdepth == 1


class _RedundantTemp(_OutermostFuncStmt):
    """First `return <expr>` (expr not already a bare Name) in the OUTERMOST
    function → `_Tk = <expr>` then `return _Tk`. Value evaluated exactly once,
    same as the original return → semantics-preserving by construction."""
    def __init__(self, temp: str):
        super().__init__()
        self.temp = temp

    def leave_SimpleStatementLine(self, o, u):
        if (self.done or not self._outermost() or len(u.body) != 1
                or not isinstance(u.body[0], cst.Return)):
            return u
        ret = u.body[0]
        if ret.value is None or isinstance(ret.value, cst.Name):
            return u  # bare `return x` / bare return → pointless temp, skip
        self.done = True
        assign = cst.SimpleStatementLine([cst.Assign(
            [cst.AssignTarget(cst.Name(self.temp))], ret.value)])
        newret = cst.SimpleStatementLine([cst.Return(cst.Name(self.temp))])
        return cst.FlattenSentinel([assign, newret])


def _is_bool_returning(expr: cst.BaseExpression) -> bool:
    """N4: Comparison, or `not <Comparison/Name/Call>` — NOT BoolOp/and/or."""
    if isinstance(expr, cst.Comparison):
        return True
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Not):
        inner = expr.expression
        return not isinstance(inner, cst.BooleanOperation)
    return False


class _BooleanExpand(_OutermostFuncStmt):
    """First `return <bool-expr>` (strict N4 precondition) in the OUTERMOST
    function → if/else."""
    def leave_SimpleStatementLine(self, o, u):
        if (self.done or not self._outermost() or len(u.body) != 1
                or not isinstance(u.body[0], cst.Return)):
            return u
        ret = u.body[0]
        if ret.value is None or not _is_bool_returning(ret.value):
            return u
        self.done = True
        return cst.If(
            test=ret.value,
            body=cst.IndentedBlock([cst.SimpleStatementLine(
                [cst.Return(cst.Name("True"))])]),
            orelse=cst.Else(cst.IndentedBlock([cst.SimpleStatementLine(
                [cst.Return(cst.Name("False"))])])),
        )


class _OutermostFuncBody(cst.CSTTransformer):
    """Base: act ONCE on the OUTERMOST FunctionDef's body.

    `leave_IndentedBlock` fires post-order, so "first block" is the innermost
    nested block — wrong target for a body-prepend (the stmt could land inside
    an `if`/`for` the tests don't reach). Tracking FunctionDef depth and
    rewriting the outermost def's `.body` is unambiguous and shape-independent.
    """
    def __init__(self):
        self.depth = 0
        self.done = False

    def visit_FunctionDef(self, node):  # noqa: N802
        self.depth += 1

    def leave_FunctionDef(self, orig, updated):  # noqa: N802
        self.depth -= 1
        if self.depth != 0 or self.done or not isinstance(
                updated.body, cst.IndentedBlock):
            return updated
        self.done = True
        return updated.with_changes(
            body=updated.body.with_changes(
                body=self._new_body(updated.body.body)))

    def _new_body(self, body):  # override
        raise NotImplementedError


class _DeadCruft(_OutermostFuncBody):
    """Prepend a literal-only `assert True` to the OUTERMOST function body.
    Purely inert (no binding; `-O` would strip it but HumanEval runs plain
    python). Always executed → no behavioural effect, shape-independent."""
    def _new_body(self, body):
        return [cst.SimpleStatementLine([cst.Assert(cst.Name("True"))]), *body]


class _InjectComment(_OutermostFuncBody):
    """Leading `# ...` comment on the first statement of the OUTERMOST body.
    Comment nodes carry no semantics."""
    def _new_body(self, body):
        if not body:
            return body
        first = body[0]
        lead = (*first.leading_lines,
                cst.EmptyLine(comment=cst.Comment("# compute the result")))
        return [first.with_changes(leading_lines=lead), *body[1:]]


AXIS2 = {
    "redundant_temp": lambda src: cst.parse_module(src).visit(
        _RedundantTemp(_reserved_temp(src))).code,
    "boolean_expand": lambda src: cst.parse_module(src).visit(_BooleanExpand()).code,
    "dead_cruft":     lambda src: cst.parse_module(src).visit(_DeadCruft()).code,
    "inject_comment": lambda src: cst.parse_module(src).visit(_InjectComment()).code,
}


# --- negative control (TEST/CANARY ONLY — deliberately semantics-breaking) ---
# Never in AXIS2 / never in the real menu. Sole purpose: prove the HumanEval
# validate() backstop is wired up and rejects a broken transform.
#
# It MUST be a *universal, always-applicable, guaranteed* break — not a
# code-shape-dependent edit. (An earlier "flip the first comparison" control
# was abandoned: on bodies with no comparison it is a silent no-op → false
# "pass", and on HumanEval/154 a real comparison flip STILL passed the unit
# tests — concrete evidence that validate() is a backstop, not a correctness
# oracle. This is precisely why every *real* transform is semantics-preserving
# by construction and unit-tested for executional equivalence; validate() is
# only the secondary net for rare libcst edge cases on the full build.)

class _RaiseInjector(_OutermostFuncBody):
    """Prepend `raise AssertionError('neg-control')` to the OUTERMOST function
    body. The function now raises on EVERY call → every unit test fails, for
    every HumanEval problem, independent of code shape. Guaranteed break."""
    def _new_body(self, body):
        return [cst.SimpleStatementLine([cst.Raise(cst.Call(
            func=cst.Name("AssertionError"),
            args=[cst.Arg(cst.SimpleString("'neg-control'"))]))]), *body]


def _neg_control(src: str) -> str:
    return cst.parse_module(src).visit(_RaiseInjector()).code

NEG_CONTROL = {"_neg_control_raise": _neg_control}


# --- top-level pipeline -----------------------------------------------------

def stylize(question: str, v2_answer: str, scheme: str | None,
            axis2: tuple[str, ...] = (), negative_control: bool = False) -> dict:
    """Return {answer, meta} or raises. answer is in v2 (line-1-col-0) format.
    Caller is responsible for HumanEval re-validation + revert (the backstop).

    scheme=None → no rename (mapping empty); used to measure an Axis-2
    transform's ΔlogP in isolation, and for the libcst no-op round-trip check.
    negative_control=True → applies the quarantined semantics-breaking
    transform (canary only; MUST fail HumanEval validation).
    """
    full_src, params, _ = build_full_func_src(question, v2_answer)
    ast.parse(full_src)  # parse guard (reference `parse_skipped` fallback)
    if scheme is None:
        mapping: dict[str, str] = {}
        renamed = cst.parse_module(full_src).code  # libcst round-trip, no rename
    else:
        mapping = compute_rename_map(full_src, params, scheme)
        renamed = apply_rename(full_src, mapping)
        if mapping and not verify_occurrence_equivalence(full_src, mapping, renamed):
            raise ValueError("occurrence-equivalence check failed (N3)")
    src = renamed
    applied = []
    for t in axis2:
        new = AXIS2[t](src)
        if new != src:
            applied.append(t)
            src = new
    if negative_control:
        src = next(iter(NEG_CONTROL.values()))(src)
    cst.parse_module(src)  # must still parse
    return {
        "answer": to_v2_format(src),
        "meta": {"scheme": scheme, "mapping": mapping,
                 "axis2_requested": list(axis2), "axis2_applied": applied,
                 "negative_control": negative_control},
    }
