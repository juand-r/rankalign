"""Shared helpers for the inventory builders (training + eval).

Single source of truth for model/task normalization and setting classification,
so _build_training_inventory.py and _build_eval_inventory.py agree.
"""
from __future__ import annotations

import re

SETTING_NAME = {
    "s1": "SFT-lo", "s2": "RankAlign", "s3": "New+fsx", "s4": "New+fsx+tc",
    "s5": "RankAlign+fsx+tc", "s6": "RankAlign+tc", "s7": "New+fsx+negtc",
    "s8": "RankAlign+fsx+negtc", "s9": "RankAlign+negtc", "s10": "RankAlign+fsx",
    "s11": "New+tc", "s12": "New+negtc", "s13": "SFT+cft",
}


def norm_model(m: str) -> str:
    m = m.strip("-")
    aliases = {
        "g4-31b": "gemma-4-31B-it",
        "gemma-4-31b-it": "gemma-4-31B-it",
        "gemma2-9b-it": "gemma-2-9b-it",
        "gemma2-2b-it": "gemma-2-2b-it",
        "gemma2-2b": "gemma-2-2b",
        "Qwen--Qwen3.5-9B": "qwen3.5-9b",
        "Qwen3.5-9B": "qwen3.5-9b",
        "qwen3.5-9b": "qwen3.5-9b",
    }
    return aliases.get(m, m)


def norm_task(t: str) -> str:
    t = t.strip("-")
    if t.startswith("membership"):
        return "membership"
    if t.startswith("persona"):
        return "persona"
    if t.startswith("ifeval"):
        return "ifeval"
    if "correct-upper" in t or t == "cu":
        return "humaneval-cu"
    if "correct-multi" in t or t == "cm":
        return "humaneval-cm"
    if t.startswith("ambigqa"):
        return "ambigqa"
    if "hc-b2d" in t or "hypernym" in t:
        return "hypernym"
    if "rosch" in t:
        return "rosch"
    return t


def eval_task_to_train_task(eval_task: str) -> str:
    """Map an EVAL task name to the training task it tests."""
    if eval_task.startswith("rosch"):
        return "membership"
    if eval_task.startswith("persona"):
        return "persona"
    if eval_task.startswith("ifeval"):
        return "ifeval"
    if "humaneval-v2.1correct-upper" in eval_task:
        return "humaneval-cu"
    if "humaneval-v2.1correct-multi" in eval_task:
        return "humaneval-cm"
    return eval_task


def classify_setting(tokens: set[str], explicit_s: str | None = None) -> str:
    """Return s1..s13 (or a descriptive label) from flag tokens or explicit sN.

    Follows SETTINGS_REFERENCE.md. Recognizes both long (`pref0.0`, `nllv1.0`,
    `force-same-x`, `tc-self`) and abbreviated (`p0`, `nv1`, `fsx`, `tcs`) forms.
    """
    if explicit_s:
        return explicit_s if explicit_s.startswith("s") else "s" + explicit_s
    has = lambda *xs: any(x in tokens for x in xs)  # noqa: E731
    pref0 = has("pref0.0", "p0")
    nll = has("nllv1.0", "nv1", "nllg1.0", "ng1")
    cft = has("cft")
    fsx = has("force-same-x", "fsx")
    tcs = has("tc-self", "tcs")
    tcn = has("tc-neg", "tcn")
    if pref0:  # SFT family
        return "s13" if cft else "s1"
    if nll:  # comb family
        if fsx:
            return "s4" if tcs else "s7" if tcn else "s3"
        return "s11" if tcs else "s12" if tcn else "comb-notc-nofsx?"
    # pref-only / RankAlign family
    if fsx:
        return "s5" if tcs else "s8" if tcn else "s10"
    return "s6" if tcs else "s9" if tcn else "s2"


def setting_sort_key(s: str):
    m = re.match(r"s(\d+)", s)
    return (0, int(m.group(1))) if m else (1, s)
