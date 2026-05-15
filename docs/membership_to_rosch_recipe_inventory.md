# membership-sans-rosch-v0 → rosch — knob inventory and what's missing

This doc inventories every training recipe we've evaluated on the
membership-sans-rosch-v0 → rosch transfer pipeline, organized by the
three knobs we currently care about for ablation:

- **TC**: `none` / `tc-self` / `tc-neg` (offline TC during training)
- **fsx** (`force-same-x`): preference pairs only between same-prompt
  rows (here: same membership category)
- **vlo** (`vallogodds`): pass `--validator-log-odds` so the validator
  scores used for pair selection during training are log-odds rather
  than raw probabilities

This doc only considers **preference-only-loss runs** (i.e. true
RankAlign-family variants with `--preference_loss_weight=1`,
`--nll_validator_weight=0`, `--nll_generator_weight=0`). Variants
that mix in NLL on validator/generator (`nllv1.0_nllg1.0`), or that
flip to pure SFT (`pref0.0_nllv1.0_nllg1.0`), or that include
semi-supervision (`semi0.1`) or label-only (`labelonly0.1`) are
**deliberately excluded** here. We can revisit those after isolating
the three knobs above.

## Naming convention going forward

The 9-variant launchers
(`scripts/run_train_membership_quickiter.sh`,
`scripts/run_train_rosch_online_quickiter.sh`) bake `--force-same-x`
into both the `PREF_BASE` and `SFT_BASE` argument blocks. So **every
variant in those launchers — including the one we've been calling
"RankAlign baseline" — is actually `RankAlign + force-same-x`**, not the
plain RankAlign of the original paper.

To keep this honest in plots, tables, and prose:

| Old label | New label |
| --- | --- |
| RankAlign baseline | **RankAlign+fsx** |
| SFT (NLL all) | **SFT+fsx** |
| RankAlign + offline self-TC | RankAlign+fsx + offline self-TC |
| RankAlign + online self-TC | RankAlign+fsx + online self-TC |
| RankAlign + online pair selection | RankAlign+fsx + online pairs |
| ... etc. | ... etc. |

Plain **RankAlign** (no fsx, no TC, no vlo, pref-loss only) is a recipe
we have **not** trained on membership. It is the missing reference
point for several of the comparisons we'd like to make.

## Inventory matrix (gemma-2-2b, epoch2, pref-loss-only)

3 knobs × {none, tc-self, tc-neg} × {fsx Y/N} × {vlo Y/N} = 12 cells.

|   | TC=none | TC=self | TC=neg |
| --- | --- | --- | --- |
| **fsx=N, vlo=N** | ⚠ have **+semi** only: `full-completion_semi0.1` (May 2; **self ref only**) | ❌ missing | ❌ missing |
| **fsx=Y, vlo=N** | ✅ `full-completion_force-same-x` (May 13) — RankAlign+fsx | ✅ `tc-self_full-completion_force-same-x` (May 13) | ✅ `tc-neg_full-completion_force-same-x` (May 13) |
| **fsx=N, vlo=Y** | ❌ missing | ❌ missing | ❌ missing |
| **fsx=Y, vlo=Y** | ❌ missing | ⚠ have **+semi** only: `tc-self_full-completion_force-same-x_vallogodds_semi0.1` (May 2; **self ref only**) | ⚠ have **+semi** only: `tc-neg_full-completion_force-same-x_vallogodds_semi0.1` (May 2; **neg ref only**) |

Key:

- **✅** = clean cell, trained without any extra flags, evaluated on all 10 rosch tasks at epoch2 across self/neg/basetyp[neg] eval refs.
- **⚠ have +semi only** = trained, but with the additional `semi0.1` flag baked in. So the cell isn't a clean one-knob change vs the surrounding row — semi is co-varying. May-2 cohort eval also only ran a single eval-ref per row (whichever matched the TC choice).
- **❌ missing** = no training run for this combination.

So we have **3 / 12 clean pref-loss-only cells filled**, plus **3 contaminated-with-semi cells**. 6 cells are completely empty.

### Plain-RankAlign approximation: `full-completion_semi0.1`

There IS a no-fsx pref-only run on disk: `full-completion_semi0.1`
(May 2). It's `pref=1`, `nllv=0`, `nllg=0`, no fsx, no TC, no vlo, but
**semi-supervised label loss with weight 0.1**. Its self-ref gen-ROC
on the 10 rosch tasks averages **80.82 (9.40)**.

Compared to RankAlign+fsx (no semi, self-ref) at **81.63 (8.14)**, this
is a 2-knob delta:

- **A − F = +0.81** ≈ "what fsx adds, while we strip semi at the same
  time." This is *not* an isolated fsx effect.

The honest version of "what does fsx add?" requires a `(fsx=N, TC=none,
vlo=N, semi=N)` run — i.e. **plain RankAlign without semi** — which
we don't have.

## What this inventory says about the user's questions

| Question | Answerable from data on disk? |
| --- | --- |
| Does RankAlign+fsx beat plain RankAlign (effect of fsx)? | **No, cleanly.** Closest is **+0.81** but that's `(fsx, no semi) − (no fsx, +semi)` — fsx and semi flip together. |
| Does RankAlign+fsx+TC-self beat plain RankAlign? | **No, cleanly.** Closest is **+5.88**, but folds in fsx flipping ON and semi flipping OFF on top of adding TC. Need a plain-RankAlign training run to isolate. |
| Does RankAlign+fsx+vlo+TC beat plain RankAlign? | **No, twice over** — missing both plain RankAlign and the clean (fsx,vlo,TC,no-semi) cell. |
| What does vallogodds add (one-knob change)? | **No** — no pair on disk that differs only in vlo. The only vlo runs always also have semi, and the no-vlo runs don't have semi. |
| What does force-same-x add (one-knob change)? | **No** — same problem; the only no-fsx run also has semi. |
| Does RankAlign+fsx+TC beat RankAlign+fsx? | **Yes**, this is the +5.07 measurement (paired-bootstrap version is +4.74). May 13 cohort. |
| Does +vlo+semi help on top of fsx+TC? | **Hurts on the self side (−5.39), neutral-to-slightly-positive on the neg side (+1.19).** But +vlo and +semi are confounded. |

## Headline numbers we can already compute

Gen-ROC × 100, mean(std) across 10 rosch tasks, gemma-2-2b epoch2,
pref-loss-only. From
[`outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md`](../outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md).

| Recipe | TC | fsx | vlo | semi | gen-ROC self | gen-ROC neg |
| --- | --- | --- | --- | --- | --- | --- |
| **(approx plain RankAlign)** `+semi` only | none | N | N | Y | 80.82 (9.40) | — (not run) |
| RankAlign+fsx | none | Y | N | N | 81.63 (8.14) | 82.96 (9.11) |
| RankAlign+fsx + offline self-TC | tc-self | Y | N | N | 86.70 (6.94) | — |
| RankAlign+fsx + offline neg-TC | tc-neg | Y | N | N | — | 80.82 (12.87) |
| RankAlign+fsx + offline self-TC + vlo + semi | tc-self | Y | Y | Y | 81.31 (8.21) | — |
| RankAlign+fsx + offline neg-TC + vlo + semi | tc-neg | Y | Y | Y | — | 82.01 (11.55) |

A few directional reads from this table (each one a **multi-knob delta**, not an isolated effect):

- **TC self vs no TC, both with fsx (no vlo, no semi):** 86.70 − 81.63 = **+5.07** — clean (this is the May 13 +4.74 effect, slightly different rounding on different scoring of typcorr-vs-self ref).
- **+vlo +semi together on top of fsx + tc-self:** 81.31 − 86.70 = **−5.39** (joint vlo+semi is bad on top of TC-self, not isolated).
- **+vlo +semi together on top of fsx + tc-neg:** 82.01 − 80.82 = **+1.19** (joint vlo+semi is mildly positive on top of TC-neg).
- **fsx vs no-fsx, both pref-only with one extra knob (semi vs none):** 81.63 − 80.82 = **+0.81** (∼0; this is fsx−semi, not fsx alone).
- **TC self vs approx-plain (across both fsx and semi flips):** 86.70 − 80.82 = **+5.88** (this is the closest we can get to "does TC beat plain RankAlign", but it folds in fsx flipping ON and semi flipping OFF).

## Three (or four) new training runs that would close the gap

All in pref-loss-only land (`--preference_loss_weight 1
--nll_validator_weight 0 --nll_generator_weight 0`), no semi, no
labelonly.

1. **Plain RankAlign** = `fsx=N, TC=none, vlo=N`.
   This is the canonical missing reference point. Pairs with
   RankAlign+fsx (existing) to isolate **fsx** alone, and is the LHS
   of "does RankAlign+anything beat plain RankAlign?".

2. **RankAlign+fsx+vlo** = `fsx=Y, TC=none, vlo=Y`.
   Pairs with the existing RankAlign+fsx cell to isolate **vallogodds**
   (one-knob change), and pairs with NEW #1 to test "does
   fsx+vlo beat plain RankAlign?".

3. **RankAlign+fsx+vlo+TC-self** = `fsx=Y, TC=self, vlo=Y`.
   Pairs with NEW #2 to isolate **TC on top of fsx+vlo**, pairs with
   the existing RankAlign+fsx+TC-self to isolate **vlo on top of
   fsx+TC**, and pairs with NEW #1 to give the "does the kitchen-sink
   recipe (fsx+vlo+TC) beat plain RankAlign?" answer the user asked
   for.

4. (**Optional**) **RankAlign+TC-self** = `fsx=N, TC=self, vlo=N`.
   Pairs with NEW #1 to ask "does TC alone (without fsx) help?" — i.e.
   isolates whether the TC benefit is fsx-dependent.

With NEW 1+2+3, the matrix below covers the questions the user asked:

|   | TC=none | TC=self |
| --- | --- | --- |
| **fsx=N, vlo=N** | ✅ NEW #1 | (still missing — only needed if we want fsx-effect-on-TC; that's the optional NEW #4) |
| **fsx=Y, vlo=N** | ✅ have | ✅ have |
| **fsx=Y, vlo=Y** | ✅ NEW #2 | ✅ NEW #3 |

### Comparison map after adding NEW #1, #2, #3

| Question | Comparison | Available after the new runs? |
| --- | --- | --- |
| Effect of **fsx** (on top of plain RankAlign, no TC) | RankAlign+fsx (have) − Plain RankAlign (NEW #1) | ✅ |
| Effect of **vlo** (on top of fsx, no TC) | RankAlign+fsx+vlo (NEW #2) − RankAlign+fsx (have) | ✅ |
| Effect of **TC** (on top of fsx, no vlo) | RankAlign+fsx+TC-self (have) − RankAlign+fsx (have) | ✅ already have it (+4.74 / +5.07) |
| Does fsx+TC beat plain RankAlign? | RankAlign+fsx+TC-self (have) − Plain RankAlign (NEW #1) | ✅ |
| Does fsx+vlo+TC beat plain RankAlign? | RankAlign+fsx+vlo+TC-self (NEW #3) − Plain RankAlign (NEW #1) | ✅ |
| Effect of **vlo** *on top of TC+fsx* (does vlo amplify or shrink the TC bonus?) | RankAlign+fsx+vlo+TC (NEW #3) − RankAlign+fsx+TC (have) | ✅ |
| Does TC alone (no fsx) help? | RankAlign+TC-self (NEW #4) − Plain RankAlign (NEW #1) | optional, requires NEW #4 |

Three runs is the minimum to answer all five user questions.

## Across-cohort caveats

When comparing the May 2 cohort (`outputs/`) and the May 13 cohort
(`outputs-quickiter/`):

- **Both used `gemma-2-2b` for the relevant pref-loss rows**, so model
  identity isn't a confound.
- **Both used `delta=0.15`, 3 epochs, the `-all` task variant**, so
  training-data quantity matches.
- **Different score-CSV column orderings, but same underlying metric**
  computation. Both go through `scripts/summarize_scores_file.py` →
  `gen_score_typcorr` → ROC-AUC vs binary `label`.
- **Different commit / different `eval_by_claude.py` versions** (May 2
  predates several refactors). Tiny numeric differences possible at the
  3rd decimal but not at the level of the +4.74 effect.

If we want maximally apples-to-apples comparisons, the new training
runs above should be done on the same git commit as the May 13 launcher,
so that the only thing varying is the training flags themselves.
