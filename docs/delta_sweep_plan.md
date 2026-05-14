# AmbigQA delta sweep plan (RankAlign baseline only, gemma-2-2b)

Companion to [ambigqa_validator_calibration_analysis.md](ambigqa_validator_calibration_analysis.md).

Drafted 2026-05-13.

## The question

Does RankAlign's gen-ROC on ambigqa-train-as-test rise with `delta` (as
validator label noise drops) before falling (as pair quantity becomes the
binding constraint)? Two existing data points so far:

| delta | RankAlign baseline gen-ROC self (× 100) |
| --- | --- |
| 0.15 | 71.82 |
| 1.0 | 61.07 |

The single-point comparison "0.15 > 1.0" is consistent with two very different
stories:

1. **Quantity dominates throughout.** RankAlign monotonically *decreases* with
   delta. There is no peak above 71.82. The calibration argument is dead on
   this task at this sample budget, and we should look elsewhere
   (more-data, GT pairs, better validator) rather than fiddle with delta.
2. **There is a peak in the middle.** RankAlign rises with delta over some
   range (where dropping the noisiest pairs helps more than losing data
   hurts), then falls. Calibration matters; we just picked the wrong delta
   the first time.

A 3-point sweep between 0.15 and 1.0 distinguishes these two stories.

## Setup

**Identical to the existing delta=0.15 and delta=1.0 RankAlign baselines except for `--delta`:**

- model = `google/gemma-2-2b`
- task  = `ambigqa-train-as-test`, all (5,110 pairs sampled per epoch)
- recipe: `--train_g_or_d g --num_epochs 3 --save_steps 999 --force-same-x`
- loss: pure preference (`--preference_loss_weight 1 --nll_validator_weight 0 --nll_generator_weight 0`)
- validator: `--validator-log-odds`
- no TC (RankAlign baseline only — TC adds confounds we untangle in a later
  experiment, not in this sweep)

Each run produces an `epoch2` checkpoint of the form:

```
v6-google--gemma-2-2b-deltaD-epoch2--ambigqa-train-as-test-all--d2g--random--alpha1.0--full-completion--force-same-x
```

(D ∈ {0.3, 0.5, 0.7}.)

**Cost per point:** ~3.5 h training + ~1 h eval (4 TC refs) on 1 GPU.

## Sweep points

Chosen to align with the bucket boundaries in the calibration table from the
companion doc:

| delta | yes-vs-no pairs kept | est. accuracy of kept set | note |
| --- | --- | --- | --- |
| 0.15 (already done) | 41,564 | 58.8% | bottom |
| 0.3  (sweep) | 26,833 | ~62% | drop the chance-level bucket only |
| 0.5  (sweep) | 16,838 | ~71% | drop two lowest buckets; balanced regime |
| 0.7  (sweep) | ~10,400 | ~76% | mid-bucket; resolves bucket-edge vs smooth-curve |
| 1.0 (already done) | 4,022 | ~92% | top (also the original prediction) |

Why exactly 3 intermediates: 2 is the minimum to detect a peak between two
endpoints (4 total points), and adding a 3rd at 0.7 lets us see whether the
curve is sensitive to bucket boundaries vs interpolating smoothly. More than
3 is diminishing returns relative to the ~4-h-per-point cost.

## Eval matrix per checkpoint

Same as `scripts/run_eval_ambigqa_2b_delta1.sh` (and the matched-task
delta=0.15 baseline): self / neg / basetyp / basetypneg, `--validator-log-odds`.

For the headline curve we only need `self` and `neg`, but the other two are
cheap and let the new points slot into the canonical 4-table layout if/when
we run TC variants at the chosen "best" delta (see follow-ups below).

**The existing delta=1.0 eval (job 38009) timed out before basetyp/basetypneg
were written; we should rerun those two so the matrix is complete for the
record.** Same script; just sbatch with --time=2:00:00 instead of 1:00:00.

## Pre-registered decision rules

What we conclude from each plausible curve shape:

| pattern | conclusion | next step |
| --- | --- | --- |
| Monotonically decreasing in delta (0.15 best) | quantity dominates; calibration argument dead on this task / sample budget | drop the validator-quality direction; pivot to GT-pair training (`--ground-truth-not-validator`) or better validator |
| Peak strictly between 0.15 and 1.0 | calibration matters; we just picked the wrong delta | re-run the 9-variant TC matrix (rosch-style) at the peak delta; that's where prediction #4 of the calibration doc actually gets its real test |
| Tie / flat (within ~1.5 gen-ROC pp) | the trade-off is shallow; neither direction is clearly winning | low priority to revisit; main story is just "ambigqa is hard for this size of validator" |
| Non-monotonic and noisy (e.g. dips at 0.3, peaks at 0.5, dips at 0.7) | seed sensitivity dominates; sweep budget too small to see signal | repeat with multiple seeds before drawing conclusions |

We commit to these *before* seeing the results so we don't post-hoc-rationalize
the curve.

## Plumbing

### Scripts
- New launcher: `scripts/run_train_ambigqa_2b_deltasweep.sh`
  - Submits 3 training jobs (delta ∈ {0.3, 0.5, 0.7}) via `run 1 4 ...`
  - Records JIDs in `overnight/ambigqa_deltasweep_jobids.txt`
- Eval launcher: clone `run_eval_ambigqa_2b_delta1.sh` × 3, parameterized by
  delta, OR a single fan-out launcher submitting 3 sbatch jobs with
  `--dependency=afterany:<train JIDs>` and `--time=2:00:00` (so we don't
  re-time-out like 38009).
- Summary: a small script that reads the 5 RankAlign-baseline gen-ROC numbers
  (delta ∈ {0.15, 0.3, 0.5, 0.7, 1.0}) from outputs-quickiter/ and emits a
  markdown table + a 1-line conclusion based on the decision rules above.

### Total compute
- Training: 3 × ~3.5 h = ~10.5 GPU-h (parallelizable)
- Eval:     3 × ~1 h    = ~3 GPU-h
- Plus: 1 × eval-completion job for delta=1.0 basetyp/basetypneg (~1 h)
- Wall clock if all 3 + 1 run in parallel: ~5 h

## What this sweep deliberately does *not* test

- **TC variants.** The sweep is RankAlign-only on purpose. Mixing TC into
  the sweep confounds the calibration question with the offset-reweighting
  question. If the sweep finds a peak, *then* we run the 9-variant
  TC matrix at that peak delta to test prediction #4 of the calibration
  doc. If the sweep is monotonically decreasing, we don't run TC variants
  at all (no useful peak to test from).
- **2b-it.** The companion doc already showed 2b-it's validator is
  miscalibrated *across all `|Δv|` buckets* — raising delta does not help
  there. Running the sweep on 2b-it is wasted compute; that variant needs
  the GT-pair experiment instead.
- **Held-out test.** Train==test memorization probe still applies. The
  sweep tells us about the matched-task setup; cross-task generalization
  needs a different experiment.

## Follow-ups conditional on the sweep result

Run *only* if a peak appears at some delta D* ∈ {0.3, 0.5, 0.7}:

1. **9-variant TC matrix at D\*.** Same recipe as the rosch-fb / membership
   training launchers, but on ambigqa-train-as-test with `--delta D*`.
   This is the first place prediction #4 actually gets tested under the
   right operating point.
2. **Same-budget bigger-data attempt at D\*.** Re-run with `total_samples`
   bumped to e.g. 20,000 to test whether the data-quantity effect is the
   binding constraint at D*; if so, we know how to spend more compute.
3. **Cross-task held-out probe.** Once a winner emerges from the matched
   sweep, evaluate that checkpoint on a held-out ambigqa split (not
   train-as-test) to see whether the matched gain transfers. If it
   doesn't, the whole sweep was a memorization story and we revise.

These follow-ups are *not* in the current plan budget — they're decision
points after the sweep results land.
