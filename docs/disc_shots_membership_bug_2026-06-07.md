# disc-shots bug: qwen membership/rosch trained+evaluated zero-shot (should be few-shot)

**Found & fixed 2026-06-07.** This note documents a discriminator-shots (`disc_shots`)
inconsistency in the RankAlign experiments: **Qwen3.5-9B membership/rosch models were trained
*and* evaluated with `disc_shots=zero`, while gemma-2-9b-it membership used `few`.** Membership
should be **few**. The qwen membership models (original paper runs *and* the first wandb-rerun
batch) are therefore wrong on this axis and are being re-run with `few`.

## Background: what disc_shots should be

`disc_shots` controls few-shot vs zero-shot prompting of the **discriminator** ("is this a
member? Yes/No"). It is set two independent places:

- **Training** (`ranking_loss_ref_fix.py`): default `None` → auto-detect. The code
  (lines ~378–393) sets `disc_shots="zero"` for **instruct** models (`-it`, or Qwen3+
  post-trained) and `"few"` for **base** models, unless `--disc-shots` overrides it.
- **Eval** (`eval_by_claude.py`): argparse default is **`few`** (line ~2106), used directly
  with **no** instruct→zero auto-detect.

The intended per-task methodology (from `run_gemma2_cell.sh`, the reference cell launcher):

| Task | disc_shots (train) | disc_shots (eval) | Notes |
|------|--------------------|-------------------|-------|
| **membership / rosch** | **few** | **few** | explicit `--disc-shots few` override (instruct models auto-detect to zero, so the override is REQUIRED) |
| **persona** | few | few | same |
| **ifeval** | zero | zero | `make_prompt_ifeval` does not implement few-shot — zero is correct for *all* models |

## The bug

The qwen launchers (`run_qwen35_cell.sh`, `run_qwen35_v7b_cell.sh`, and the first version of
`run_qwen35_cell_mll.sbatch`) **never passed `--disc-shots`** for training and hardcoded
`--disc-shots zero` for eval. Since Qwen3.5 auto-detects to `zero`, **qwen membership ran
zero-shot disc for both train and eval** — inconsistent with gemma-2-9b-it membership (`few`).

This affects:
- the **original paper** qwen membership models (the `latkes/rankalign-v7-qwen3.5-9b-membership-*`
  HF repos), and
- the **first wandb-rerun batch** of qwen membership (run 2026-06-06, jobs 43938–43943).

ifeval qwen is **unaffected** (zero is correct for ifeval). gemma models are unaffected.

## Verification (reproducible)

All checked against real artifacts on 2026-06-07, not inferred from scripts.

**1. Original qwen membership = zero (the bug).** The trainer's stdout is bundled in each qwen HF
model repo as `training_log.log.gz`:
```python
from huggingface_hub import hf_hub_download; import gzip
p = hf_hub_download("latkes/rankalign-v7-qwen3.5-9b-membership-s7-ep2", "training_log.log.gz")
# grep -> "Detected instruct model (Qwen3+ post-trained)" ; "gen_shots: zero, disc_shots: zero"  (NO override)
```

**2. gemma-2-9b-it membership = few (correct), both train and eval.**
- Train — actual wandb run `output.log` (20 finished runs, May-24 delta-bins batch):
  `Overriding disc_shots to: few`.
- Eval — pairing each `disc_shots: <few|zero>` log line with the `Detailed scores saved to:
  scores_…<model>…<task>….csv` it produced, across `/datastor2/jdr/logs/*.out`:
  ```bash
  for f in /datastor2/jdr/logs/*.out; do LC_ALL=C awk '
    /disc_shots: few/{cur="few"} /disc_shots: zero/{cur="zero"}
    /Detailed scores saved/ && /gemma-2-9b-it/ && /rosch/ {print cur}' "$f"; done | sort | uniq -c
  #  -> 490 few   0 zero
  ```
  Sanity: the same attribution on `/Qwen/ && /rosch/` returns **20 zero** — correctly catches the
  qwen bug and validates the method. (File-level grep is unreliable: one log holds many evals; e.g.
  log 42505 matched "rosch" but is actually a gemma-2-**2b** *persona* eval at `few`.)

## Fix

`scripts/run_qwen35_cell_mll.sbatch` now sets `DISC_SHOTS` **per task** and threads it into BOTH the
train and eval calls (commit on `longform`, 2026-06-07):
- membership/persona → `--disc-shots few`
- ifeval → `--disc-shots zero`

Validated live: the corrected jobs log `Overriding disc_shots to: few`.

## Actions taken (2026-06-07)

1. **Archived** the 30 wrong (zero) qwen membership rerun dirs:
   `models2-rerun-wandb/` → `models2-rerun-wandb-WRONG-disczero-membership/` (with a `README.txt`).
2. **Relaunched** the 5 qwen membership cells (s1/s2/s3/s4/s7) with the fixed launcher —
   **train + eval**, `disc_shots=few`, wandb runs tagged `…-membership-s*-discfew`, models back into
   `models2-rerun-wandb/`. mll jobs **43949–43953** (launched ~17:25 CT).
3. Supervisor (`scripts/monitor_qwen35_rerun.sh`) resubmits via `.monitor/resubmit_export`
   = `ALL,WANDB_SUFFIX=-discfew` (no `TRAIN_ONLY`, so resubmits keep train+eval).

## Still open (paper-level)

- The **original paper qwen membership models** (`latkes/...-membership-*` HF repos) used zero.
  The corrected `few` reruns supersede them — once cell C finishes, upload the corrected models +
  rosch scores to HF and mark/replace the old repos.
- qwen **ifeval** is fine (zero). gemma everything is fine.
- s13 was skipped for the qwen reruns (separate gap).

## Summary table

| Model × task | train | eval | status |
|---|---|---|---|
| gemma-2-9b-it × rosch/membership | few | few | ✅ correct (wandb exists) |
| qwen3.5-9b × rosch/membership (original + 1st rerun) | zero | zero | ❌ bug → superseded |
| qwen3.5-9b × rosch/membership (corrected rerun, jobs 43949–53) | **few** | **few** | 🔄 running |
| any × ifeval | zero | zero | ✅ correct (few-shot not implemented for ifeval) |
