# Debugging plan: why TC / online variants don't beat RankAlign

Captured 2026-05-11 after the rosch + ambigqa quick-iter overnight runs.
Keep this living document up to date as experiments land.

## Observation

From the four `outputs-quickiter/*_quickiter_summary*.md` tables (all numbers
× 100, generator ROC-AUC, best across eval-time TC reference per row):

| setting | RankAlign baseline (abs) | best non-SFT non-RankAlign (abs) | Δ vs baseline |
|---|---|---|---|
| rosch 2b | 86.81 | 95.75 (online self-TC) | **+8.94** |
| rosch 2b-it | 88.87 | 93.63 (offline self-TC) | **+4.76** |
| ambigqa 2b | 71.82 | 66.77 (online pairs) | **−5.05** |
| ambigqa 2b-it | 77.07 | 78.48 (online pairs) | +1.41 |

Note: this is the same data the `_debug_view.py` script extracts from the
markdown summaries; numbers are identical, just rescaled to percentage points.

So the picture is **not** "TC never helps" — TC clearly helps on rosch 2b and
helps modestly on rosch 2b-it. The real puzzle is that **TC hurts on ambigqa
2b and is roughly flat on ambigqa 2b-it**.

The single most striking difference is the **base-model validator quality**:

| | base validator ROC | RankAlign-trained validator ROC |
|---|---|---|
| rosch 2b | 96.05 | 91.09 |
| rosch 2b-it | 96.16 | 96.05 |
| ambigqa 2b | 56.09 | 58.29 |
| ambigqa 2b-it | 58.84 | 62.15 |

On ambigqa the validator is barely above chance — and yet the entire training
signal is the validator's pair ordering (filtered by `delta = v_pos − v_neg
> 0.15`). This is the lead hypothesis below.

## Lead hypothesis

**The validator's noisy ordering on ambigqa contaminates the training pairs,
and TC reweights pairs by typicality — which is uncorrelated with whether
the pair's label is correct — so TC amplifies the noise.**

This predicts:

1. On ambigqa, a large fraction of selected training pairs has the
   "wrong-sign" winner (validator picks the GT-no item over the GT-yes item).
2. SFT works because it ignores the validator entirely (it trains on
   ground-truth `yes`/`no` labels from the CSV directly).
3. Replacing validator-sourced pair labels with ground-truth labels on
   ambigqa should close the gap to SFT.

If the lead hypothesis is wrong, possible alternates:

- **A:** TC is correctly directing gradients, but on ambigqa most pairs
  already saturate `sigmoid(Δ_θ)` so the gradient is vanishingly small — TC
  just shuffles tiny updates.
- **B:** The metric is misleading. With train≡test, SFT trivially memorizes;
  the gap to RankAlign+TC may collapse on a held-out test set.
- **C:** Online TC has a training-time instability that doesn't show up on
  rosch (small / repeated pairs) but does on ambigqa (large / each pair seen
  ≈once).

## Plan, in increasing order of effort

Status legend: `[ ]` not started, `[~]` running / WIP, `[x]` done.

### Tier 0 — sanity checks (free, no retraining)

Cheap diagnostics over the existing eval CSVs and saved data. These should
either confirm or kill the lead hypothesis without burning GPU.

- `[x]` **#1. Pair-label correctness.** **DONE.** Reproduce the training-time
  pair construction offline: group items by prompt (`force-same-x`), form all
  within-group pairs, filter by `|val_score_i − val_score_j| > 0.15`, then
  ask: among pairs where the two items have *different* ground-truth labels
  (i.e. one yes, one no — these are the only pairs where GT has a preference),
  what fraction does the validator order correctly? Result: rosch 97.82% /
  97.03%, ambigqa 58.82% / 58.53% — lead hypothesis strongly confirmed.
- `[x]` **#2. TC weight vs ground-truth correctness.** **DONE** as part of
  the same script. Pairs with extreme tc_adj (positive or negative) have very
  different correctness rates on ambigqa (31% at tc_adj < −2 vs 82% at
  tc_adj > 2), so TC weight IS a quality signal — but it's unclear that
  train-time TC actually exploits this; defer to follow-up.
- `[x]` **#3. Validator calibration plot.** **DONE.** Per-`|Δv|`-bucket
  accuracy reveals a sharp 2b-vs-2b-it asymmetry on ambigqa: 2b's validator
  is well-calibrated (raising `delta` could fix it), 2b-it's is miscalibrated
  across all confidence levels (raising `delta` won't help). Detail:
  [ambigqa_validator_calibration_analysis.md](ambigqa_validator_calibration_analysis.md).

Single deliverable: `scripts/diag_pair_label_correctness.py` writes
[diag_pair_correctness_report.md](diag_pair_correctness_report.md) with all
three breakdowns for the four (model, task) combos.

### Tier 1 — controlled retrains (1 GPU each, hours)

- `[~]` **#3a. (NEW) Rerun ambigqa 2b with `delta = 1.0`.** Suggested by the
  Tier 0 calibration result — on 2b the validator is well-calibrated, so
  raising the pair-confidence filter should give us ~92%-correct pairs
  (instead of the current 58.82%). Submitted as Slurm job 37613 via
  `scripts/run_train_ambigqa_2b_delta1.sh`. If gen-ROC jumps materially
  toward SFT, the validator-quality hypothesis is confirmed and we'll rerun
  TC variants under `delta = 1.0` next.
- `[ ]` **#4. Run with `--ground-truth-not-validator` on ambigqa 2b-it.** The
  2b-it validator is miscalibrated, so `delta`-tuning can't help; only GT
  labels can. The flag exists in `scripts/ranking_loss_ref_online.py`. If
  gen-ROC under GT-pairs jumps near SFT, we've localized the failure to
  validator-sourced labels (not RankAlign or TC themselves).
- `[ ]` **#5. Rosch with deliberately weakened validator.** Inject label
  noise into the validator-side pair labels on rosch (e.g. flip 30% of
  winners). Should reproduce the ambigqa pattern. Confirms the mechanism
  is "validator quality, not domain".
- `[ ]` **#6. Bump online-pair-selection cadence.** Currently re-elects only
  at epoch boundary (so once for 3 epochs). Re-elect every k=200 steps so
  the validator's mid-training improvements actually feed back. Combine
  with re-electing against the *current generator's* current ordering, so
  pairs the model already orders correctly drop out — this is the part
  where "online" should actually matter.

### Tier 2 — instrumentation we'd want either way

- `[ ]` **#7. Per-step held-out gen-ROC / val-ROC.** Log them every N steps,
  not just at epoch boundary. Lets us see whether TC overshoots and decays
  vs underperforms throughout. Without intra-epoch curves we can't tell.
- `[ ]` **#8. Fraction of pairs with non-saturated gradient.** With
  `--force-same-x` and `delta=0.15`, on ambigqa most pairs may already have
  `Δ_θ` saturating sigmoid — no learning signal. Log
  `mean(sigmoid'(Δ_θ))` per step as a "live gradient" indicator.

### Tier 3 — break the memorization confound

- `[ ]` **#9. Held-out test set on ambigqa.** Train==test (current setup)
  trivially favors SFT. Re-run on a real held-out split; the picture above
  may flip qualitatively.

## How this connects to the existing research plan

The lead hypothesis dovetails with the "live-with-grads TC" discussion in
[IMPORTANT-RESEARCH-PLAN.md](IMPORTANT-RESEARCH-PLAN.md): if the validator
is the bottleneck on ambigqa, *no* train-time TC variant (offline reweighting
or online gradient steering) can fix it, because TC operates on the model's
own logprobs and never touches the validator. The training-time signal
itself is corrupt. That's why **#4** (replace validator with GT) is the
single most diagnostic experiment — it tells us whether the rest of the TC
research program is even pointed at the right bottleneck.
