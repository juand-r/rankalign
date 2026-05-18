"""analyze_canary.py — per-transform ΔlogP table + keep/cut + degenerate stop.
CPU only. Joins each variant to its row's `original` IN-RUN (no external
baseline). Writes CANARY_DELTA_LOGP.md.

Decision rule (non-circular — matches red_team_brief §5 / BUILD_PLAN):
  Metric = LENGTH-NORMALIZED ΔlogP(y|x) (per-token nats = sum_logp/n_tok).
  Raw summed ΔlogP is length-confounded (transforms change token count) and
  is REPORTED for context only, never gated — matching the codebase's
  established lenorm-dominant finding (humaneval-v2/V2_ANALYSIS_REPORT).
  KEEP a transform iff
    (a) mean per-token ΔlogP(y|x) ≤ −THRESH AND per-instance sign consistent
        (≥ SIGN_FRAC of instances negative), AND
    (c) revert rate < REVERT_MAX.
  (b) human-plausibility is eyeballed from the emitted samples (flagged).
  ΔlogP(y), Δ(TC)=Δcond−Δuncond (per-token), raw-sum: REPORTED ONLY, never gate.
  Borderline (sign straddles) → CUT (conservative; bump sample to revisit).
  If NO transform is kept → "PREMISE NOT SUPPORTED — DO NOT BUILD".
  --thresh is in per-token nats; the report prints the full per-unit
  distribution so the bar can be re-set from data post-hoc.
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
    ap.add_argument("--thresh", type=float, default=0.05,
                    help="material |mean per-token ΔlogP(y|x)| bar (nats/token; "
                         "conservative default — re-set from the report's "
                         "distribution + sign-consistency post-hoc)")
    ap.add_argument("--sign-frac", type=float, default=0.70)
    ap.add_argument("--revert-max", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=15,
                    help="below this applied-N, the decision is flagged low-N "
                         "in the report (not auto-overridden — human reviews)")
    args = ap.parse_args()

    pairs = {r["variant_id"]: r for r in _load(args.pairs)}
    scores = {s["variant_id"]: s for s in _load(args.scores)}

    # ΔlogP baseline per (task,row) = the NOOP variant's scores, NOT the raw
    # `original`. The noop goes through the identical
    # build_full_func_src+to_v2_format pipeline as every transformed variant,
    # so any column-0-trailing-code re-indent (inherited from the trusted
    # reference) is COMMON to baseline and variants and CANCELS in Δ. Joining
    # to raw `original` would conflate the transform with a whole-block
    # indentation shift on the rows where the model emitted trailing code.
    base = {}
    for vid, r in pairs.items():
        if r["label"] == "noop" and vid in scores:
            base[(r["task_id"], r["row_idx"])] = scores[vid]

    # attempted (incl reverts) and scored deltas, per unit.
    # CRITICAL (round-3 fix): the keep/cut metric is *length-normalized*
    # ΔlogP(y|x) = sum_logp/n_tok (per-token nats). Raw summed ΔlogP confounds
    # "more atypical" with "longer" — every transform changes token count
    # (verbose ~+30 tok, inject_comment/dead_cruft ~+5), so a sum-based bar
    # would KEEP length-padding transforms purely for adding tokens, the exact
    # opposite of the design intent (BUILD_PLAN: cut length/noise) and counter
    # to this codebase's established finding that lenorm is the dominant axis
    # (humaneval-v2/V2_ANALYSIS_REPORT). Raw-sum is still REPORTED for context.
    attempted = defaultdict(int)
    reverts = defaultdict(int)
    dcond = defaultdict(list)        # per-token Δ (the keep/cut metric)
    duncond = defaultdict(list)      # per-token Δ (reported only)
    dtc = defaultdict(list)          # per-token Δ(TC) (reported only)
    dcond_sum = defaultdict(list)    # raw summed Δcond (context only)
    samples = defaultdict(list)
    noop_skipped = defaultdict(int)   # transform requested but a structural
    #                                   no-op on this row (precondition unmet)
    dropped_unscored = 0

    def _applied(r, kind, name) -> bool:
        """Did the unit actually change the code on this row? (from stylize
        meta). A structural no-op (e.g. boolean_expand on a row with no bare
        comparison return) must NOT dilute the unit's keep/cut stats."""
        if kind == "axis2":
            return name in (r.get("axis2_applied") or [])
        if kind == "scheme":
            return bool(r.get("scheme_applied"))
        return True

    for vid, r in pairs.items():
        kind, name = _unit(r)
        if kind is None:
            continue
        key = f"{kind}:{name}"
        attempted[key] += 1
        if not r.get("validated"):
            reverts[key] += 1
            continue
        if not _applied(r, kind, name):
            noop_skipped[key] += 1  # excluded from Δ/sign; applicability shown
            continue
        o = base.get((r["task_id"], r["row_idx"]))
        s = scores.get(vid)
        if not o or not s:
            dropped_unscored += 1  # logged; affects effective decision N
            continue
        # per-token (length-normalized) — the decision metric
        pc = s["sum_logp_cond"] / s["n_tok"] - o["sum_logp_cond"] / o["n_tok"]
        pu = s["sum_logp_uncond"] / s["n_tok"] - o["sum_logp_uncond"] / o["n_tok"]
        dcond[key].append(pc)
        duncond[key].append(pu)
        dtc[key].append(pc - pu)
        dcond_sum[key].append(s["sum_logp_cond"] - o["sum_logp_cond"])
        if len(samples[key]) < 2:
            samples[key].append((r["task_id"], r["row_idx"]))

    lines = ["# CANARY_DELTA_LOGP — per-transform ΔlogP (gemma-4-31B-it)\n"]
    lines.append(
        f"Decision metric = **length-normalized** ΔlogP(y|x) (per-token "
        f"nats); raw-sum reported for context only (length-confounded — not "
        f"gated on). KEEP iff mean per-token ΔlogP(y|x) ≤ −{args.thresh} AND "
        f"≥{args.sign_frac:.0%} instances negative AND revert "
        f"<{args.revert_max:.0%}. ΔlogP(y)/Δ(TC) reported, never gating.\n")
    lines.append(
        "Stats are over rows where the transform ACTUALLY applied (structural "
        "no-ops excluded so they don't dilute Δ/sign — applicability shown "
        "separately). ΔlogP baseline = the libcst-noop variant (same pipeline "
        "→ re-indent artifact cancels).\n")
    if dropped_unscored:
        lines.append(f"NOTE: {dropped_unscored} validated+applied variants "
                     f"dropped (row's noop baseline unscored) — reduces N.\n")
    lines.append("| unit | n_used | noop | revert% | meanΔcond/tok | "
                 "medΔcond/tok | %neg | meanΔcond_sum | meanΔunc/tok | "
                 "meanΔTC/tok | decision |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    kept = []
    for key in sorted(attempted):
        n = len(dcond[key])
        rev = reverts[key] / attempted[key] if attempted[key] else 1.0
        noop = noop_skipped.get(key, 0)
        if n == 0:
            why = ("no rows where it applied" if noop else "no data")
            lines.append(f"| {key} | 0 | {noop} | {rev:.0%} | — | — | — | "
                         f"— | — | — | CUT ({why}) |")
            continue
        mc, md = st.mean(dcond[key]), st.median(dcond[key])
        pneg = sum(1 for x in dcond[key] if x < 0) / n
        mu, mt = st.mean(duncond[key]), st.mean(dtc[key])
        msum = st.mean(dcond_sum[key])
        keep = (mc <= -args.thresh and pneg >= args.sign_frac
                and rev < args.revert_max)
        dec = "KEEP" if keep else "CUT"
        if n < args.min_n:  # self-defending: flag small-N verdicts
            dec += f" (low-N, n={n} — applicability {n}/{n + noop})"
        if keep:
            kept.append(key)
        lines.append(f"| {key} | {n} | {noop} | {rev:.0%} | {mc:+.3f} | "
                     f"{md:+.3f} | {pneg:.0%} | {msum:+.1f} | {mu:+.3f} | "
                     f"{mt:+.3f} | {dec} |")

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

    # Machine-readable menu the full builder consumes (reproducible loop:
    # canary measures → kept_menu.json → build_v2_1_correct_multi reads it).
    import json as _json
    menu = {
        "kept": kept,  # e.g. ["scheme:upper","axis2:redundant_temp",...]
        "kept_schemes": [k.split(":", 1)[1] for k in kept
                         if k.startswith("scheme:")],
        "kept_axis2": [k.split(":", 1)[1] for k in kept
                       if k.startswith("axis2:")],
        "thresh_per_tok": args.thresh, "sign_frac": args.sign_frac,
        "revert_max": args.revert_max, "min_n": args.min_n,
        "premise_supported": bool(kept),
    }
    menu_path = Path(args.out).parent / "kept_menu.json"
    menu_path.write_text(_json.dumps(menu, indent=2))
    print("\n".join(lines[-12:]))
    print(f"\n-> {args.out}\n-> {menu_path}")


if __name__ == "__main__":
    main()
