# Standard table format for membership / rosch / ambigqa quick-iter results

This doc fixes a single, comprehensive way of presenting the 9-variant
results so we don't have to re-derive the conventions every time. Keep it
in sync with `scripts/run_train_membership_quickiter.sh` and
`scripts/run_train_rosch_online_quickiter.sh`.

## What we are measuring

For every (training task, eval task, model) we run the **same 9 variants**
and report 4 metrics per cell:

- **Generator ROC-AUC** (the headline — does the model rank positives
  above negatives among the items it might generate?)
- Validator ROC-AUC
- Validator accuracy at threshold 0
- Pearson(generator score, validator score)

All numeric cells are scaled by **× 100** (percentage points for ROC /
accuracy; 0–100 units for Pearson). Multi-task aggregates report
**mean (std)** across tasks. Single-task aggregates omit the std.

## The 4-table layout

For each metric (most importantly Generator ROC-AUC) we emit **4 tables**
per (training task, model):

| # | Table | Eval ref | Rows | Cols |
| --- | --- | --- | --- | --- |
| 1 | Baselines (self) | self | Base HF, SFT | gen ROC |
| 2 | TC × pairs (self) | mixed (see below) | offline TC, online TC, no TC | offline pairs, online pairs |
| 3 | Baselines (neg) | neg | Base HF, SFT | gen ROC |
| 4 | TC × pairs (neg) | mixed (see below) | offline TC, online TC, no TC | offline pairs, online pairs |

Tables 1 / 3 hold the "comparators": the untrained base model, and the
SFT-trained one. They use the eval ref that matches the table title
(self in 1, neg in 3).

Tables 2 / 4 hold the **2-axis controlled-grid** of preference-trained
variants. The two axes are:

- **TC**: how the typicality correction is computed. `none` (no
  correction), `offline` (subtract a precomputed `log P_base(y|·)`,
  frozen), `online` (subtract a live `log P_θ(y|·)` with gradients).
- **pairs**: how training pairs are chosen. `offline` (filtered once at
  the start using base-validator scores, frozen for all epochs);
  `online` (re-elected per epoch using the *current* model's validator).

So Tables 2 and 4 are a clean **3 (TC) × 2 (pairs) grid of preference
runs that share every other flag**. RankAlign baseline IS the (no TC,
offline pairs) cell.

### Eval ref convention for Tables 2 / 4 (mixed-but-best-per-row)

Each TC row uses the eval ref that *matches* its training-time
typicality assumption:

| TC row | Eval ref used |
| --- | --- |
| offline TC | `basetyp` (or `basetypneg` in Table 4) — same frozen-base typicality term that was subtracted at train time |
| online TC | `self` (or `neg` in Table 4) — current model's own typicality |
| no TC | `self` (or `neg` in Table 4) — there was no typicality term to be consistent with, so use the standard ref |

This is empirically a wash for offline self-TC (basetyp vs self ≈ same
within ±1 pt, sometimes self is even better) but it matters for offline
neg-TC (basetypneg pairs naturally with the neg-TC training objective)
and the online variants — neg variants in particular collapse hard
under `basetypneg` (~45 vs ~78 under `neg`).

## Variant ↔ cell map

The 9 variants emitted by the launcher map onto the cells as follows.
"Self side" populates Tables 1+2; "neg side" populates Tables 3+4.

### Tables 1 + 2 (self side)

| Table | Row | Variant id | Variant label |
| --- | --- | --- | --- |
| 1 | Base HF | 0 | Base HF (untrained gemma-2-2b) |
| 1 | SFT | 6 | SFT (NLL all) |
| 2 | (offline TC, offline pairs) | 2 | RankAlign + offline self-TC |
| 2 | (offline TC, online pairs) | — | **never trained** |
| 2 | (online TC, offline pairs) | 3 | RankAlign + ONLINE self-TC |
| 2 | (online TC, online pairs) | 5 | RankAlign + ONLINE self-TC + ONLINE pair selection |
| 2 | (no TC, offline pairs) | 1 | **RankAlign baseline** |
| 2 | (no TC, online pairs) | 4 | RankAlign + ONLINE pair selection |

### Tables 3 + 4 (neg side)

| Table | Row | Variant id | Variant label |
| --- | --- | --- | --- |
| 3 | Base HF | 0 | Base HF |
| 3 | SFT | 6 | SFT (NLL all) |
| 4 | (offline TC, offline pairs) | 7 | RankAlign + offline neg-TC |
| 4 | (offline TC, online pairs) | — | **never trained** |
| 4 | (online TC, offline pairs) | 8 | RankAlign + ONLINE neg-TC |
| 4 | (online TC, online pairs) | 9 | RankAlign + ONLINE neg-TC + ONLINE pair selection |
| 4 | (no TC, offline pairs) | 1 | **RankAlign baseline** (re-used) |
| 4 | (no TC, online pairs) | 4 | RankAlign + ONLINE pair selection (re-used) |

Note that the (no TC) row of the grid uses the same two checkpoints
(variants 1 and 4) on both the self and the neg side — only the eval ref
differs. There is no neg-side analogue of variants 1 and 4 because
"no TC" is symmetric across the two eval refs.

## Exact flags per variant

Common to all 8 preference-trained variants (1, 2, 3, 4, 5, 7, 8, 9):

```
--train_g_or_d g
--split_type random
--num_epochs 3
--delta 0.15
--save_steps 999          (only epoch 0 + final epoch are saved by default)
--all
--force-same-x            (pairs only within same generator prompt)
--nll_validator_weight 0
--nll_generator_weight 0
--preference_loss_weight 1
--no-upload-hf
```

`--validator-log-odds` is **NOT** passed by any of the 9 variants (i.e.,
it defaults to False). Eval scripts always pass `--validator-log-odds`.
This is a deliberate train/eval link-function choice for the validator
score; for g-mode preference training the flag is essentially inert
because the score path uses `log P(y|x)` regardless. It only meaningfully
changes behavior in the SFT validator-NLL loss (variant 6) — see
"Caveats" below.

The flags that vary across the 8 PREF variants are exactly:

| variant | label | `--self-typicality` | `--neg-typicality` | `--online-typicality` | `--online-pair-selection` |
| --- | --- | --- | --- | --- | --- |
| 1 | RankAlign baseline | — | — | — | — |
| 2 | + offline self-TC | yes | — | — | — |
| 3 | + ONLINE self-TC | yes | — | yes | — |
| 4 | + ONLINE pair selection | — | — | — | yes |
| 5 | + ONLINE self-TC + ONLINE pair selection | yes | — | yes | yes |
| 7 | + offline neg-TC | — | yes | — | — |
| 8 | + ONLINE neg-TC | — | yes | yes | — |
| 9 | + ONLINE neg-TC + ONLINE pair selection | — | yes | yes | yes |

Variant 6 (SFT) uses an entirely different loss recipe:

```
--nll_validator_weight 1
--nll_generator_weight 1
--preference_loss_weight 0
(... rest as PREF_BASE ...)
```

It is not part of the 3 × 2 grid.

## Loss formulas (g-mode, preference-trained variants)

Let `s_θ(y|x) = log P_θ(y | x_g)` be the generator-score for completion
`y` under prompt `x_g` (as computed by `sum_completion_logprobs`).
Let `t(y) = log P_·(y | x_typ)` be the typicality term, computed under
either the **frozen base** model (offline TC) or **the current model θ**
(online TC), and using either the standard typicality prompt
(`--self-typicality`) or the negated prompt (`--neg-typicality`).

For each pair (winner i, loser j):

```
score_i = s_θ(y_i | x_i)  −  α · t(y_i)        (α=0 if no TC, else 1)
score_j = s_θ(y_j | x_j)  −  α · t(y_j)
loss     = −log σ( score_i − score_j )
```

In offline TC, `t(y)` is a precomputed scalar baked into the dataset
(no gradient w.r.t. θ — purely a per-pair reweighting). In online TC,
`t(y)` is computed by an additional forward pass through the current
model with gradients, so the typicality term **does** flow gradient
direction through θ.

`--online-pair-selection` orthogonally controls the **pair pool**: with
the flag, before each epoch the validator scores are re-evaluated under
the current model and pairs are re-filtered by `|Δv| > delta`. Without
it, the pair pool is fixed at the initial (base-model) validator pass.

## Caveats — what the table does NOT control

1. **No global RNG seed.** The training script never calls
   `torch.manual_seed`, `np.random.seed`, or sets `args.seed` for run
   reproducibility. Only the dataset-split seed (`seed=0`) is fixed.
   Across the 8 PREF runs the model init weights are deterministic
   (loaded from a fixed HF checkpoint) and the initial pair set is
   deterministic (since validator scores at t=0 are deterministic given
   the same model), but **batch shuffling and any stochastic ops differ
   across runs**. So Tables 2/4 cells include some run-to-run noise on
   top of the actual flag effects.

2. **`--validator-log-odds` train/eval mismatch in SFT only.** Eval
   always uses log-odds; training default is log-probs. For g-mode
   preference variants this is irrelevant (the pref loss path doesn't
   use the flag). For SFT, the validator-NLL term is trained against
   `−log P(Yes|x)` (a log-prob) but evaluated against
   `log P(Yes|x) − log P(No|x)` (a log-odds), so the SFT row of Tables
   1/3 has an additional link-function inconsistency that the
   pref-trained rows do not have.

3. **Memorization confound on memorization-probe training tasks.**
   For tasks like `rosch-furniture-and-bird` and `ambigqa-train-as-test`
   we use train==test, so SFT trivially wins on the headline metric.
   For `membership-sans-rosch-v0 → rosch` evaluation, the train and eval
   tasks differ by name but have substantial item-level overlap (see
   `BUCKETED_by_overlap_*.md`), so item-overlap-bucketed reports are
   the primary comparison rather than the global mean.

4. **`epoch0` vs `epoch2` checkpoints.** With `--save_steps 999` the
   training script only saves the model at epoch index 0 (after the 1st
   training pass) and at the final epoch. Mid-run intermediates
   (`epoch1`-on-disk for a 3-epoch run) are not saved unless the run
   uses `--save_steps 1`. When comparing tables, always check which
   epoch the model dirs encode.

## Where the build scripts live

- Training launcher: `scripts/run_train_membership_quickiter.sh`,
  `scripts/run_train_rosch_online_quickiter.sh`
- Eval launchers: `scripts/launch_membership_to_rosch_evals.sh`,
  `scripts/launch_rosch_crosstask_evals.sh`
- Aggregation + per-task tables (old 4-column layout):
  `scripts/build_quickiter_summary_tables.py`
- MEAN + bucketed tables (old layout): currently
  `scripts/build_membership_after1ep_to_rosch_buckets.py` and
  `scripts/build_membership_to_rosch_buckets.py`. **These need to be
  updated to emit the 4-table format described here**; until then,
  generate the 4-table layout by hand from the long-form CSV.

## Quick checklist before publishing a results table

- [ ] All 8 preference variants share PREF_BASE (no extra flags slipped
      in).
- [ ] Pair-set construction is deterministic at t=0 → cells differ only
      in the (TC, pairs) axes (modulo RNG noise on shuffling).
- [ ] Eval ref per row matches the convention above (offline TC →
      basetyp; online TC and no TC → self/neg).
- [ ] Numbers scaled by ×100, mean (std) over tasks.
- [ ] Note which checkpoint epoch is being read (`epoch0` vs `epoch2`).
- [ ] Note any train==test confound (memorization probe).
