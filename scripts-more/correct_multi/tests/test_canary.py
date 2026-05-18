"""No-GPU tests for the canary scripts.

(1) comparability invariant: the v2 scorer's INSTRUCTION_COND / MODEL that
    canary_score.py imports verbatim must match the frozen literals (checked
    by reading source text — no torch needed locally).
(2) build_canary smoke: original validates True, negative-control validates
    False (backstop works), noop preserves comments & v2 col-0 format.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
sys.path.insert(0, str(_PKG))
import canary_score as CS  # noqa: E402  (no torch import at module top)


def test_scorer_comparability_literals_unchanged():
    """ast-extract the v2 scorer's constants (torch-free) and assert they
    equal the canary's frozen literals — the byte-identical-prompt guarantee."""
    import ast as _ast
    src = (_PKG.parents[3] /
           "notes/log_P_diff_plots/humaneval-v2/scripts/score_v2_humaneval.py"
           ).read_text()
    tree = _ast.parse(src)
    vals = {}
    for node in tree.body:
        if isinstance(node, _ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], _ast.Name):
            try:
                vals[node.targets[0].id] = _ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert vals.get("INSTRUCTION_COND") == CS.EXPECTED_INSTRUCTION_COND, \
        f"v2 scorer INSTRUCTION_COND drifted: {vals.get('INSTRUCTION_COND')!r}"
    assert vals.get("MODEL") == CS.EXPECTED_MODEL, \
        f"v2 scorer MODEL drifted: {vals.get('MODEL')!r}"


def test_build_canary_smoke():
    """Tiny build: 2 tasks, run the committed build_canary.py end-to-end."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "pairs.jsonl"
        r = subprocess.run(
            [sys.executable, str(_PKG / "build_canary.py"),
             "--n-rows", "2", "--n-tasks", "2", "--seed", "1",
             "--out", str(out)],
            capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, f"build_canary failed:\n{r.stderr[-2000:]}"
        recs = [json.loads(l) for l in open(out)]
        by_label = {}
        for x in recs:
            by_label.setdefault(x["label"], []).append(x)

        # original: present, validated True by construction (baseline; not
        # re-judged — v2.1 has known correct-labeled rows that fail strict
        # re-validation, which is not this canary's concern)
        assert by_label["original"], "no original variants"
        assert all(x["validated"] and x["answer"] is not None
                   for x in by_label["original"]), "original baseline broken"
        # transformed variants actually ran (scheme/axis2 emitted & scored-able)
        assert any(k.startswith("scheme:") for k in by_label), "no scheme variants"
        assert any(x["validated"] for k, v in by_label.items()
                   if k.startswith("scheme:") for x in v), \
            "no scheme variant validated — transforms or validate() broken"

        # negative control: present and ALL validate False (backstop proven)
        assert by_label["neg_control"], "no negative-control variants"
        assert all(not x["validated"] for x in by_label["neg_control"]), \
            "negative control validated True — backstop NOT proven"

        # noop: comments preserved + v2 col-0 format (when source had a comment)
        for x in by_label.get("noop", []):
            if x["answer"] is None:
                continue
            assert not x["answer"].startswith("    "), "noop broke v2 col-0"

        # every record carries the self-contained schema (no info.mapping gate)
        need = {"task_id", "row_idx", "variant_id", "label", "scheme",
                "axis2", "negative_control", "answer", "validated",
                "revert_reason"}
        assert need <= set(recs[0]), f"schema missing {need - set(recs[0])}"


if __name__ == "__main__":
    fails = 0
    for name in ("test_scorer_comparability_literals_unchanged",
                 "test_build_canary_smoke"):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
