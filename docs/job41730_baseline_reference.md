# Job 41730 — baseline reference for ppd / shape-budget-mode comparisons

## Why this exists

41730 is a fix1 g-mode training on `membership-sans-rosch-v0` with the
**old** (pre-`--per-prompt-delta`, pre-`--shape-budget-mode`) regime. We
deliberately let it finish so we have a concrete trained checkpoint
against which to compare the two new mechanisms.

## Config

| Flag                       | Value                                  |
|---                          |---                                     |
| Model                      | `google/gemma-2-2b`                    |
| Task                       | `membership-sans-rosch-v0`             |
| Loss                       | `comb` (nll_v=1, nll_g=1, pref=1)      |
| Mode                       | `g`                                    |
| `--semi-supervised`        | `0.1`                                  |
| `--validator-log-odds`     | ON                                     |
| `--self-typicality`        | ON                                     |
| `--force-same-x` (fsx)     | **ON**                                 |
| `--delta-bins`             | **10** (auto-delta = 0.1875)           |
| `--per-prompt-delta` (ppd) | **OFF** (single global delta)          |
| `--shape-budget-mode`      | **per-prompt** (default at launch)     |

Submitted: 2026-05-23 22:21 (US/Central). Started: 23:35.
Wall-time limit: 5h. Slurm logs: `/datastor2/jdr/logs/41730.{out,err}`.
Models dir: `./models2/` (resolves to `/datastor1/.../models2/` — see
`.cursor/rules/save-models-to-datastor2.mdc` for why this is bad; future
runs go to `/datastor2`).

## Pair construction stats (from `~/logs/41730.out` lines 2086–2102)

```
Prompts: 68 group(s) (fsx on)
|L+| = 85  |L-| = 88  |U| = 1895  (of 2068 total)
  case_A    :     1023 valid pairs
  mixed_neg :        0 valid pairs
  mixed_pos :        0 valid pairs
  both_U    :    24642 valid pairs
  total     :    25665

Sampled per shape (aggregated across prompts):
  case_A    :    428/    1023  (weight=0.2)
  both_U    :   4685/   24642  (weight=0.4)
Total sampled: 5113/5110
```

So the actual sampled supervised/unlabeled mix is ~8.4% / ~91.6% —
pinned to the labeled-prompt fraction (semi=0.1), NOT to the user's
shape weights (0.2 / 0.4). This is the regime that
`--shape-budget-mode global` was designed to fix.

## What to do once it finishes

1. **Eval** the saved model (use the standard membership-sans-rosch eval
   pipeline) and record the score files alongside the future
   ppd-ON / global-ON variants.
2. **Inspect the per-run JSON log** at `models2/training_run_logs/*.json`
   (if it was written — that block was added in commit e6aa14a2).
   Confirm the recorded global delta = 0.1875, etc.
3. **Compare against** future runs that flip ONE flag at a time:
   - 41730 baseline: ppd OFF, mode=per-prompt
   - +ppd:           ppd ON,  mode=per-prompt
   - +global:        ppd OFF, mode=global
   - +both:          ppd ON,  mode=global
4. The cleanest way to get those four cells is to launch the three new
   variants with identical other flags (split-seed, total_samples, etc.)
   so 41730 acts as a true reference cell. Document the cells in
   `docs/comb_loss_g_mode_concerns.md` or a new comparison doc.

## Caveats

- Models dir is on /datastor1; if the volume saturates before save the
  reference is lost. Plan B: as soon as the job finishes (and *before*
  any other heavy /datastor1 write), `cp -r` the saved checkpoint dir to
  /datastor2 to insulate it.
- This was launched directly by the user (not via the `run` sbatch
  wrapper from the agent), so the exact CLI is reconstructed from the
  log header and the in-script diagnostic prints, not from a saved
  command line.
