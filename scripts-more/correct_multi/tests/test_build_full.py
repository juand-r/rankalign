"""Fast CPU smoke tests for build_v2_1_correct_multi.py (no GPU, no full
82-task run). Covers: seeded-assignment determinism/range, stylize+validate
on a real v2.1 correct row, and the premise-not-supported ABORT path."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))
import build_v2_1_correct_multi as B  # noqa: E402

SCHEMES = ["upper", "cryptic", "verbose"]
AXIS2 = ["redundant_temp", "dead_cruft", "inject_comment"]


def test_assign_deterministic_and_in_range():
    a = B._assign("humaneval_7", 3, SCHEMES, AXIS2, 42)
    b = B._assign("humaneval_7", 3, SCHEMES, AXIS2, 42)
    assert a == b, "assignment not deterministic for same (task,row,seed)"
    sc, ax = a
    assert sc in SCHEMES and set(ax) <= set(AXIS2)
    # different rows generally differ in selection (sanity, not strict)
    seen = {B._assign("humaneval_7", i, SCHEMES, AXIS2, 42) for i in range(20)}
    assert len(seen) > 1, "assignment ignores row_idx"


def test_stylize_with_fallback_on_real_v21_correct_row():
    B_problems = B.load_problems()
    v21 = _PKG.parents[1] / "data/humaneval/v2.1"
    csv = sorted(v21.glob("humaneval_*.csv"))[0]
    df = pd.read_csv(csv)
    cr = df[df["correct"].astype(str).str.lower() == "yes"]
    if cr.empty:                       # pick another file if first has none
        for c in sorted(v21.glob("humaneval_*.csv")):
            df = pd.read_csv(c)
            cr = df[df["correct"].astype(str).str.lower() == "yes"]
            if not cr.empty:
                csv = c
                break
    row = cr.iloc[0]
    prob = B_problems[B.slug_to_id(csv.stem)]
    ans, sc_ok, ax_ok, rev, blame = B._stylize_with_fallback(
        str(row["question"]), str(row["answer"]), "upper",
        ("redundant_temp", "inject_comment"), prob)
    ok, _ = B.validate(prob, ans)
    assert ok, "stylized-or-reverted answer must pass HumanEval validation"
    # full menu is semantics-preserving by construction → should not revert
    assert not rev, f"unexpected revert (blame={blame}) on a clean row"


def test_premise_not_supported_aborts():
    with tempfile.TemporaryDirectory() as d:
        menu = Path(d) / "kept_menu.json"
        menu.write_text(json.dumps({
            "kept": [], "kept_schemes": [], "kept_axis2": [],
            "premise_supported": False}))
        r = subprocess.run(
            [sys.executable, str(_PKG / "build_v2_1_correct_multi.py"),
             "--menu", str(menu)], capture_output=True, text=True, timeout=120)
        assert r.returncode != 0, "must ABORT when premise not supported"
        assert "premise NOT supported" in (r.stdout + r.stderr), \
            "abort message must explain the degenerate-outcome stop"


if __name__ == "__main__":
    fails = 0
    for n in ("test_assign_deterministic_and_in_range",
              "test_stylize_with_fallback_on_real_v21_correct_row",
              "test_premise_not_supported_aborts"):
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
