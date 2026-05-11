# AmbigQA validator calibration: why TC didn't help, and what to do

Companion to [DEBUGGING-PLAN.md](DEBUGGING-PLAN.md) and
[diag_pair_correctness_report.md](diag_pair_correctness_report.md).
Generated 2026-05-11.

## TL;DR

- On rosch, the base validator orders yes-vs-no training pairs **97%** correctly. RankAlign sees clean pairs, and TC variants help.
- On ambigqa, the base validator orders yes-vs-no training pairs only **58%** correctly. Trains essentially on label noise.
- The 58% number on the two ambigqa runs (2b and 2b-it) hides a sharper picture: 2b's validator is **well-calibrated** (confident pairs are reliable; the issue is just that `delta = 0.15` is too generous), while 2b-it's validator is **miscalibrated** across the board (high confidence does not predict correctness).
- Concrete cheap experiment: rerun ambigqa 2b with `delta = 1.0` instead of `0.15`. Predicted pair-correctness jumps from 58.82% → ~92%. If gen-ROC also jumps, the validator-quality hypothesis is confirmed and the rest of TC research is being bottlenecked by training pair quality on this task.

## Setup

`scripts/diag_pair_label_correctness.py` reproduces the train-time pair construction offline from each base model's eval CSV (which already contains per-item `val_score`, `gen_score`, `gen_score_typcorr` and the GT label):

1. Group items by prompt (matches `--force-same-x`).
2. Within each group, form all directed pairs (winner, loser) with `val_score_winner > val_score_loser`.
3. Filter by `|val_score_winner − val_score_loser| > delta` (training default `delta = 0.15`).
4. Among pairs where the two items have *different* GT labels (one `yes`, one `no` — these are the only pairs where the GT itself has a preference), check whether the validator-preferred item is the GT-yes one.

This faithfully reproduces what RankAlign's training data looks like at step 0 of training (before any model updates). For "online pairs" / online-TC variants the picture changes during training, but the *initial* signal RankAlign sees is exactly this.

## Headline numbers

Base-model validator pair-ordering accuracy on yes-vs-no pairs (after `|Δv| > 0.15` filter, `--force-same-x`):

| setting | items | yes-vs-no pairs | validator pair-ordering accuracy |
| --- | --- | --- | --- |
| rosch 2b | 186 | 4,180 | **97.82%** |
| rosch 2b-it | 186 | 4,380 | **97.03%** |
| ambigqa 2b | 7,992 | 41,564 | **58.82%** |
| ambigqa 2b-it | 7,992 | 75,090 | **58.53%** |

So RankAlign-on-rosch trains on near-clean preference labels; RankAlign-on-ambigqa trains on near-random labels. The difference between TC helping (rosch) and TC not helping (ambigqa) is plausibly *just* about pair quality.

## Where the noise lives — calibration

Bucket yes-vs-no pairs by `|Δv|` (the validator's confidence in its own pair ordering) and report accuracy per bucket.

### ambigqa 2b — well-calibrated, but `delta = 0.15` is too loose

| abs(Δv) bucket | pairs | validator accuracy |
| --- | --- | --- |
| 0.15 – 0.3 | 14,731 | 49.81% (chance) |
| 0.3 – 0.5 | 9,995 | 51.52% |
| 0.5 – 1.0 | 12,816 | 64.37% |
| 1.0 – 2.0 | 3,370 | **90.80%** |
| 2.0 – 4.0 | 652 | **99.69%** |

Read the column from top to bottom: as the validator becomes more confident (`|Δv|` grows), accuracy rises monotonically from 50% (chance) to ≈100%. The validator does know what it's talking about — but only on the small high-confidence tail.

The bottom three buckets (`|Δv| < 1.0`) hold **37,542 of 41,564** yes-vs-no pairs (~90% of the training data). They average around 56% accuracy. The training signal is dominated by them.

If we re-trained with `delta = 1.0` instead of `0.15`:

- pairs kept: 4,022 (down from 41,564)
- predicted accuracy on those pairs: weighted average of 90.80% × 3370 + 99.69% × 652 ≈ **92.2%**

That is essentially rosch-quality preference labels, just with much less data.

### ambigqa 2b-it — miscalibrated

| abs(Δv) bucket | pairs | validator accuracy |
| --- | --- | --- |
| 0.15 – 0.3 | 2,675 | 55.07% |
| 0.3 – 0.5 | 3,464 | 54.45% |
| 0.5 – 1.0 | 8,398 | 58.22% |
| 1.0 – 2.0 | 12,142 | 57.25% |
| 2.0 – 4.0 | 15,499 | 55.95% |
| > 4.0 | 32,912 | **61.01%** |

The accuracy column barely moves. Even pairs where the validator says `|Δv| > 4` (very strong preference) sit at only 61%. Confidence does not predict correctness here — the validator's preference signal and the ground truth signal are nearly independent. Raising `delta` does not help, because the noise is mixed in evenly at every confidence level.

### rosch (both 2b and 2b-it) — well-calibrated

For comparison, rosch 2b's table shows accuracy rising from ~80% in `(0.15, 0.3]` to ≈100% by `|Δv| > 1`. The difference between rosch and ambigqa is *not* "rosch is calibrated, ambigqa isn't" — it's "rosch has a strong validator and ambigqa has a weak one, with 2b additionally showing well-behaved calibration that we can exploit by raising `delta`".

## Why this maps onto our results

Recall the four (model, task) results (from
[the four `outputs-quickiter/*_quickiter_summary*.md` tables](../outputs-quickiter/)):

| setting | RankAlign baseline gen-ROC | best non-SFT non-RankAlign |
| --- | --- | --- |
| rosch 2b | 86.81 | **+8.94** (online self-TC, 95.75) |
| rosch 2b-it | 88.87 | **+4.76** (offline self-TC, 93.63) |
| ambigqa 2b | 71.82 | **−5.05** (online pairs, 66.77) |
| ambigqa 2b-it | 77.07 | **+1.41** (online pairs, 78.48) |

The pattern lines up: TC helps where pairs are clean (rosch), TC neutral-or-hurts where pairs are noisy (ambigqa). SFT bypasses the validator entirely (it trains directly on the GT yes/no label per item via NLL), which is exactly why SFT wins by ≈+20 on ambigqa but only ≈+10–13 on rosch.

The **2b vs 2b-it asymmetry on ambigqa** also lines up: on 2b, the validator can be salvaged by raising `delta`; on 2b-it, it cannot.

## Concrete next experiments

In priority order:

1. **Cheap, immediate:** rerun ambigqa 2b RankAlign baseline with `delta = 1.0`. If gen-ROC jumps from 71.82 toward SFT (≈92), the validator-quality hypothesis is confirmed for 2b. Script: [run_train_ambigqa_2b_delta1.sh](../scripts/run_train_ambigqa_2b_delta1.sh).
2. After #1: run the TC variants (offline self-TC, offline neg-TC) at `delta = 1.0` on 2b ambigqa. If TC helps once pairs are clean, the TC research direction is sound — it was just being starved of clean training signal.
3. **Independent of #1:** run ambigqa 2b-it with `--ground-truth-not-validator`. The 2b-it validator can't be saved by `delta` tweaks; if GT-pair training closes the gap, we've localized the failure to the validator labels rather than to RankAlign or TC themselves.
4. Long-term: replace ambigqa's validator with something better (different model, prompt, finetune on the validator task, etc.). This is a separate research thread, not a quick fix.

## What this *doesn't* prove

- We have not shown that TC + clean pairs > RankAlign + clean pairs on ambigqa. Step 2 above tests that.
- We have not shown that the 2b-it validator is irrecoverable. Step 3 tests that.
- The train==test memorization probe inflates SFT's headline numbers; on a real held-out split the relative ordering may shift.
