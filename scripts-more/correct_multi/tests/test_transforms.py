"""Unit tests for correct_multi.transforms — encode the red-team contracts.

Run: <venv-with-libcst>/python -m pytest test_transforms.py
 or: <venv-with-libcst>/python test_transforms.py   (no-pytest fallback)

The strongest check here is *executional* semantic equivalence: build a
runnable function from (sig, body), run original vs every stylized variant on
a battery of inputs, assert identical results. This is stricter than the
HumanEval backstop and catches semantics-changing transforms directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transforms as T  # noqa: E402


# (signature, v2-format body, list of arg-tuples)
# v2 format = LLM function body with line-1 dedented to col 0; lines 2+ keep
# their in-function indentation (4 sp at body level, 8 sp nested). This is
# exactly what build_humaneval_v2_1_correct_upper's re-indent expects.
CASES = [
    ("def f(n):", "result = n * 2 + 1\nreturn result", [(0,), (3,), (-4,)]),
    ("def g(xs):",
     "total = 0\n    for x in xs:\n        total += x\n    return total",
     [([],), ([1, 2, 3],), ([10],)]),
    ("def h(a, b):", "return a == b", [(1, 1), (1, 2), ("x", "x")]),
    ("def k(a, b):", "return a and b", [(1, 2), (0, 5), ([], 3)]),
    ("def m(xs):",
     "acc = [len(s) for s in xs]\n    return sorted(acc)",
     [(["a", "bb"],), ([],), (["zzz", "y"],)]),
]


def _reindent(body: str) -> str:
    return "\n".join(("    " + ln if ln.strip() and not ln.startswith("    ")
                       else ln) for ln in body.split("\n"))


def _make_fn(sig: str, v2_body: str):
    src = sig + "\n" + _reindent(v2_body)
    ns: dict = {}
    exec(compile(src, "<t>", "exec"), ns)  # noqa: S102 (test-only)
    return ns[sig.split("(")[0][4:].strip()]


def _eq_on_inputs(sig, orig_body, new_body, args_list) -> bool:
    fo, fn = _make_fn(sig, orig_body), _make_fn(sig, new_body)
    for a in args_list:
        try:
            ro = fo(*a)
        except Exception as e:  # noqa: BLE001
            ro = ("EXC", type(e).__name__)
        try:
            rn = fn(*a)
        except Exception as e:  # noqa: BLE001
            rn = ("EXC", type(e).__name__)
        if ro != rn:
            return False
    return True


AXIS2_ALL = ("redundant_temp", "boolean_expand", "dead_cruft", "inject_comment")


def test_semantics_preserved_all_schemes_and_axis2():
    """Every (scheme × each axis2 × all-axis2-stacked) preserves behaviour."""
    for sig, body, args in CASES:
        for scheme in T.RENAME_SCHEMES:
            combos = [()] + [(t,) for t in AXIS2_ALL] + [AXIS2_ALL]
            for ax in combos:
                out = T.stylize(sig, body, scheme, ax)
                assert _eq_on_inputs(sig, body, out["answer"], args), (
                    f"semantics broke: {sig} scheme={scheme} axis2={ax}\n"
                    f"--- got ---\n{out['answer']}")


def test_rename_map_respects_trusted_freeze_set():
    """Map keys ⊆ renameable; never params/_FROZEN/keyword; targets disjoint
    & valid identifiers."""
    import keyword
    for sig, body, _ in CASES:
        full, params, _ = T.build_full_func_src(sig, body)
        for scheme in T.RENAME_SCHEMES:
            m = T.compute_rename_map(full, params, scheme)
            tgts = list(m.values())
            assert len(tgts) == len(set(tgts)), f"non-injective {scheme}"
            for k, v in m.items():
                assert k not in params and k not in T._FROZEN
                assert not keyword.iskeyword(v) and v.isidentifier()
                assert v not in params and v not in T._FROZEN
    # builtins frozen: 'len'/'sorted' never renamed in case m
    full, params, _ = T.build_full_func_src(CASES[4][0], CASES[4][1])
    for scheme in T.RENAME_SCHEMES:
        m = T.compute_rename_map(full, params, scheme)
        assert "len" not in m and "sorted" not in m and "xs" not in m


def test_occurrence_equivalence_holds_and_detects_corruption():
    sig, body, _ = CASES[1]
    full, params, _ = T.build_full_func_src(sig, body)
    m = T.compute_rename_map(full, params, "upper")
    good = T.apply_rename(full, m)
    assert T.verify_occurrence_equivalence(full, m, good)
    # corrupt: rename one extra occurrence the ast reference would NOT
    corrupt = good.replace("TOTAL", "TOTAL_X", 1)
    assert not T.verify_occurrence_equivalence(full, m, corrupt)


def test_boolean_expand_only_on_real_bool():
    # h: `return a == b` — Comparison → expands
    out_h = T.stylize(CASES[2][0], CASES[2][1], "upper", ("boolean_expand",))
    assert "boolean_expand" in out_h["meta"]["axis2_applied"]
    assert "if " in out_h["answer"] and "return True" in out_h["answer"]
    # k: `return a and b` — BoolOp → MUST be a no-op (value is an operand)
    out_k = T.stylize(CASES[3][0], CASES[3][1], "upper", ("boolean_expand",))
    assert "boolean_expand" not in out_k["meta"]["axis2_applied"]
    assert _eq_on_inputs(CASES[3][0], CASES[3][1], out_k["answer"],
                         CASES[3][2])


def test_redundant_temp_fresh_vs_post_rename_identifiers():
    """cryptic emits v0,v1,…; the reserved temp must not collide (N2). Use the
    case whose return is an *expression* (`return sorted(acc)`), not a bare
    Name (bare-Name returns are intentionally skipped)."""
    sig, body, args = CASES[4]  # def m(xs): ... return sorted(acc)
    out = T.stylize(sig, body, "cryptic", ("redundant_temp",))
    assert "redundant_temp" in out["meta"]["axis2_applied"]
    assert _eq_on_inputs(sig, body, out["answer"], args)
    assert "_T" in out["answer"]  # reserved temp present, no v-collision


def test_libcst_noop_preserves_comments_and_v2_format():
    sig = "def c(n):"
    body = "# a leading comment\ntotal = n  # inline note\nreturn total"
    # no scheme effect path: pick a scheme, but assert comments survive
    out = T.stylize(sig, body, "upper", ())
    assert "# a leading comment" in out["answer"]
    assert "# inline note" in out["answer"]
    # v2 format: line 1 at column 0 (no leading 4 spaces)
    assert not out["answer"].startswith("    ")


def test_dead_cruft_and_comment_are_inert():
    for sig, body, args in CASES:
        for t in ("dead_cruft", "inject_comment"):
            out = T.stylize(sig, body, "verbose", (t,))
            assert _eq_on_inputs(sig, body, out["answer"], args), (sig, t)


def test_body_prepend_targets_outermost_func_not_nested_block():
    """Regression: leave_IndentedBlock fires post-order, so a naive 'first
    block' prepend lands in the innermost nested block (e.g. inside an `if`
    the tested inputs skip). The negative control on a body whose only
    statements are inside a conditional MUST still break ALL inputs."""
    sig = "def p(n):"
    # body where every real statement is nested under `if n > 0:`
    body = ("if n > 0:\n        result = n * 2\n        return result\n"
            "    return -1")
    args = [(5,), (-3,), (0,)]
    # negative control must fail on EVERY input (raise reaches the top level),
    # not just the n>0 branch
    out = T.stylize(sig, body, None, (), negative_control=True)
    fn = _make_fn(sig, out["answer"])
    for a in args:
        try:
            fn(*a)
            raised = False
        except AssertionError:
            raised = True
        assert raised, f"neg-control did not break input {a} (nested-block bug)"
    # dead_cruft / inject_comment stay inert even with the nested body
    for t in ("dead_cruft", "inject_comment"):
        o2 = T.stylize(sig, body, "upper", (t,))
        assert _eq_on_inputs(sig, body, o2["answer"], args), t


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
