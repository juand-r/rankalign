"""analyze_canary.py — per-transform ΔlogP table + keep/cut + degenerate stop.
CPU only. Joins each variant to its row's `original` IN-RUN (no external
baseline). Writes CANARY_DELTA_LOGP.md.

Decision rule (non-circular — matches red_team_brief §5 / BUILD_PLAN):
  KEEP a transform iff
    (a) mean ΔlogP(y|x) ≤ −THRESH  AND per-instance sign consistent
        (≥ SIGN_FRAC of instances negative), AND
    (c) revert rate < REVERT_MAX.
  (b) human-plausibility is eyeballed from the emitted samples (flagged).
  ΔlogP(y) and Δ(TC)=Δcond−Δuncond are REPORTED ONLY — never gate.
  Borderline (sign straddles) → CUT (conservative; bump sample to revisit).
  If NO transform is kept → "PREMISE NOT SUPPORTED — DO NOT BUILD".
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _load(p):
    return [json.loads(l) for l in open(p)]


def _unit(rec):
    """Map a variant to its transform 'unit' for per-transform aggregation.
    scheme:* and axis2:* are the individually-measured candidates; stacked/
    noop/neg_control are reported separately, not as keep/cut candidates."""
    lab = rec["label"]
    if lab.startswith("scheme:"):
        return ("scheme", lab.split(":", 1)[1])
    if lab.startswith("axis2:"):
        return ("axis2", lab.split(":", 1)[1])
    return (None, lab)  # original / noop / stacked:* / neg_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=5.0,
                    help="material |mean ΔlogP(y|x)| sum-nats bar")
    ap.add_argument("--sign-frac", type=float, default=0.70)
    ap.add_argument("--revert-max", type=float, default=0.10)
    args = ap.parse_args()

    pairs = {r["variant_id"]: r for r in _load(args.pairs)}
    scores = {s["variant_id"]: s for s in _load(args.scores)}

    # original per (task,row)
    orig = {}
    for vid, r in pairs.items():
        if r["label"] == "original" and vid in scores:
            orig[(r["task_id"], r["row_idx"])] = scores[vid]

    # attempted (incl reverts) and scored deltas, per unit
    attempted = defaultdict(int)
    reverts = defaultdict(int)
    dcond = defaultdict(list)
    duncond = defaultdict(list)
    dtc = defaultdict(list)
    samples = defaultdict(list)
    for vid, r in pairs.items():
        kind, name = _unit(r)
        if kind is None:
            continue
        key = f"{kind}:{name}"
        attempted[key] += 1
        if not r.get("validated"):
            reverts[key] += 1
            continue
        o = orig.get((r["task_id"], r["row_idx"]))
        s = scores.get(vid)
        if not o or not s:
            continue
        dc = s["sum_logp_cond"] - o["sum_logp_cond"]
        du = s["sum_logp_uncond"] - o["sum_logp_uncond"]
        dcond[key].append(dc)
        duncond[key].append(du)
        dtc[key].append(dc - du)
        if len(samples[key]) < 2:
            samples[key].append((r["task_id"], r["row_idx"]))

    lines = ["# CANARY_DELTA_LOGP — per-transform ΔlogP (gemma-4-31B-it)\n"]
    lines.append(f"Decision rule: KEEP iff mean ΔlogP(y|x) ≤ −{args.thresh} "
                 f"(sum nats) AND ≥{args.sign_frac:.0%} instances negative AND "
                 f"revert <{args.revert_max:.0%}. ΔlogP(y)/Δ(TC) reported only.\n")
    lines.append("| unit | n | revert | meanΔcond | medΔcond | %neg | "
                 "meanΔunc | meanΔTC | decision |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    kept = []
    for key in sorted(attempted):
        n = len(dcond[key])
        rev = reverts[key] / attempted[key] if attempted[key] else 1.0
        if n == 0:
            lines.append(f"| {key} | 0 | {rev:.0%} | — | — | — | — | — | "
                         f"CUT (no data) |")
            continue
        mc, md = st.mean(dcond[key]), st.median(dcond[key])
        pneg = sum(1 for x in dcond[key] if x < 0) / n
        mu, mt = st.mean(duncond[key]), st.mean(dtc[key])
        keep = (mc <= -args.thresh and pneg >= args.sign_frac
                and rev < args.revert_max)
        dec = "KEEP" if keep else "CUT"
        if keep:
            kept.append(key)
        lines.append(f"| {key} | {n} | {rev:.0%} | {mc:+.1f} | {md:+.1f} | "
                     f"{pneg:.0%} | {mu:+.1f} | {mt:+.1f} | {dec} |")

    # stacked / noop / neg_control diagnostics (reported, not keep/cut)
    lines.append("\n## Diagnostics (not keep/cut candidates)\n")
    for vid, r in sorted(pairs.items()):
        if r["label"] == "noop":
            same = (r.get("answer") is not None)
            lines.append(f"- noop {vid}: validated={r.get('validated')} "
                         f"(libcst round-trip; comments must survive)")
            break
    negs = [r for r in pairs.values() if r["label"] == "neg_control"]
    if negs:
        bad = sum(1 for r in negs if r.get("validated"))
        lines.append(f"- negative control: {len(negs)} emitted, "
                     f"{bad} wrongly validated (MUST be 0 — proves the "
                     f"HumanEval backstop rejects broken transforms)")

    if kept:
        verdict = f"\n## VERDICT: BUILD with kept units: {', '.join(kept)}\n"
        verdict += ("Eyeball the side-by-side samples for human-plausibility "
                    "(criterion b) before finalizing the menu.\n")
    else:
        verdict = ("\n## VERDICT: PREMISE NOT SUPPORTED — DO NOT BUILD\n"
                   "No transform clears the material-drop bar. Building from "
                   "the least-bad transforms is disallowed (degenerate-outcome "
                   "stop). Report this as the finding.\n")
    lines.append(verdict)

    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines[-12:]))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
