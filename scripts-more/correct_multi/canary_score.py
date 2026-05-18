"""canary_score.py — score logp_cond + logp_uncond for EVERY validated
variant in canary_pairs.jsonl on gemma-4-31B-it. GPU.

Reuses the v2 scorer VERBATIM (imports its functions — not a reimpl) so the
prompt path is byte-identical to v2 / correct-upper. NO `info.mapping` /
pass-gate: original AND every variant is scored (ΔlogP computed in-run by
analyze_canary.py — no external v2/v2.1 baseline join). Resumable by
variant_id.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# NOTE: torch / transformers / the v2 scorer are imported lazily inside
# main() so this module's frozen invariants (EXPECTED_*) are inspectable
# (and unit-testable) without a GPU stack installed.

# Default = the workspace-repo layout (parents[3]/notes/...). On a
# rankalign-only pod clone that path won't exist, so --v2-scripts-dir lets
# the deploy point at a STAGED copy of the exact same score_v2_humaneval.py
# (single source preserved: the startup INSTRUCTION_COND/MODEL asserts fail
# loud if the staged copy ever differs from the canonical one).
_DEFAULT_V2_SCRIPTS = (Path(__file__).resolve().parents[3] /
                       "notes/log_P_diff_plots/humaneval-v2/scripts")

# Frozen invariants — fail loud if the reused source ever drifts (the
# comparability guarantee the red-team brief requires).
EXPECTED_INSTRUCTION_COND = (
    "Complete the following Python function. "
    "Return ONLY the solution code, no markdown, starting from inside the function:"
)
EXPECTED_MODEL = "google/gemma-4-31B-it"


def _done_ids(path: str) -> set[str]:
    d = set()
    if Path(path).exists():
        for line in open(path):
            try:
                d.add(json.loads(line)["variant_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2-scripts-dir", default=str(_DEFAULT_V2_SCRIPTS),
                    help="dir containing score_v2_humaneval.py (stage the "
                         "canonical copy here for pod deployment)")
    args = ap.parse_args()

    v2dir = Path(args.v2_scripts_dir)
    if not (v2dir / "score_v2_humaneval.py").exists():
        sys.exit(f"FATAL: score_v2_humaneval.py not found in {v2dir} "
                 f"(stage the canonical copy or pass --v2-scripts-dir)")
    sys.path.insert(0, str(v2dir))
    import torch
    import score_v2_humaneval as S  # verbatim reuse (byte-identical prompts)

    assert S.INSTRUCTION_COND == EXPECTED_INSTRUCTION_COND, \
        "score_v2_humaneval.INSTRUCTION_COND drifted — comparability broken"
    assert S.MODEL == EXPECTED_MODEL, f"scorer MODEL drifted: {S.MODEL}"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {S.MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        S.MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    uncond_prefix = S.chat_prefix_ids(tok, "")  # empty-chat baseline (verbatim)
    print(f"model on {device}; uncond_prefix len={len(uncond_prefix)}", flush=True)

    rows = [json.loads(l) for l in open(args.pairs)]
    todo = [r for r in rows if r.get("validated") and r.get("answer")]
    done = _done_ids(args.out)
    print(f"{len(rows)} variants; {len(todo)} validated; "
          f"{len(done)} already scored", flush=True)

    t0 = time.time()
    n = 0
    with open(args.out, "a") as fout:
        for r in todo:
            if r["variant_id"] in done:
                continue
            try:
                sig = S.extract_signature(r["question"])
            except ValueError as e:
                print(f"  SKIP {r['variant_id']} ({e})", flush=True)
                continue
            cond_prefix = S.chat_prefix_ids(
                tok, S.build_cond_prompt(r["question"], sig))
            comp_ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
            if not comp_ids:
                print(f"  SKIP {r['variant_id']} (empty completion)", flush=True)
                continue
            lp_c, _ = S.score_completion(model, tok, cond_prefix, comp_ids,
                                         device, compute_entropy=False)
            lp_u, _ = S.score_completion(model, tok, uncond_prefix, comp_ids,
                                         device, compute_entropy=False)
            fout.write(json.dumps({
                "variant_id": r["variant_id"], "task_id": r["task_id"],
                "row_idx": r["row_idx"], "label": r["label"],
                "n_tok": len(comp_ids),
                "sum_logp_cond": float(sum(lp_c)),
                "sum_logp_uncond": float(sum(lp_u)),
            }) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                print(f"  scored {n} in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"DONE: scored {n} new variants in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
