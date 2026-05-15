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
flip to pure SFT (`pref0.0_nllv1.0_nllg1.0`), or that use label-only
(`labelonly0.1`) are **deliberately excluded** here.

> **Key insight: with pref-only weights, `--semi-supervised 0.1` is a
> mathematical no-op.** Looking at
> [`scripts/ranking_loss_ref.py:2622-2638`](../scripts/ranking_loss_ref.py)
> (and the identical block in `_online.py:2914-2930`), when
> `pref=1, nllv=0, nllg=0`:
>
> ```
> labeled_loss   = 1*preference_loss + 0*nll_v + 0*nll_g = preference_loss
> unlabeled_loss = preference_loss
> loss = pair_is_labeled * pref + (1 - pair_is_labeled) * pref = preference_loss
> ```
>
> Both branches collapse to `preference_loss`. The only side effect is
> a `_semi0.1` suffix on the model save directory. The
> `split_prompts_labeled_unlabeled` helper uses a private `random.Random(seed)`,
> so it doesn't even perturb the global RNG state.
>
> So **a pref-only May-2 run with `--semi-supervised 0.1` is identical
> to one with the flag dropped**. Treat `+semi0.1` cells as clean
> (no semi confound) for pref-loss-only ablations.

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

Plain **RankAlign** (no fsx, no TC, no vlo, pref-loss only) is the
reference point for several of the comparisons we'd like to make. We
**do** have a checkpoint for it on disk, even though it was originally
trained with `--semi-supervised 0.1` — that flag is mathematically
inert under pref-only weights (see the box above). So when this doc
refers to "Plain RankAlign" it means the on-disk
`models/v6-google--gemma-2-2b-delta0.15-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--semi0.1`
checkpoint.

## Inventory matrix (gemma-2-2b, epoch2, pref-loss-only)

3 knobs × {none, tc-self, tc-neg} × {fsx Y/N} × {vlo Y/N} = 12 cells.

|   | TC=none | TC=self | TC=neg |
| --- | --- | --- | --- |
| **fsx=N, vlo=N** | ✅ `full-completion_semi0.1` (May 2) — Plain RankAlign  *eval refs: self only* | ❌ missing | ❌ missing |
| **fsx=Y, vlo=N** | ✅ `full-completion_force-same-x` (May 13) — RankAlign+fsx  *eval refs: all 4* | ✅ `tc-self_full-completion_force-same-x` (May 13)  *eval refs: self, basetyp* | ✅ `tc-neg_full-completion_force-same-x` (May 13)  *eval refs: neg, basetypneg* |
| **fsx=N, vlo=Y** | ❌ missing | ❌ missing | ❌ missing |
| **fsx=Y, vlo=Y** | ❌ missing | ✅ `tc-self_full-completion_force-same-x_vallogodds_semi0.1` (May 2)  *eval refs: self only* | ✅ `tc-neg_full-completion_force-same-x_vallogodds_semi0.1` (May 2)  *eval refs: neg only* |

Key:

- **✅** = checkpoint trained, evaluated on all 10 rosch tasks at epoch2 (eval refs listed in italics).
- **❌ missing** = no training run for this combination.

So we have **6 / 12 cells filled** (semi=0.1 in the May 2 cells is a no-op for pref-only training, see above). 6 cells are still completely empty.

The **eval-ref coverage is uneven** across the May 2 cells — they were
each evaluated under a single ref matched to the TC choice, not all 4
refs. The May 2 checkpoints are still on disk under `models/` (verified
2026-05-14), so re-running missing eval refs is cheap (a few minutes
per (task, ref) instead of a full retrain).

## What this inventory says about the user's questions

(Numbers below are gen-ROC × 100, mean across 10 rosch tasks, gemma-2-2b epoch2, evaluated under the matching eval ref. All deltas are pref-loss-only and "semi" never enters as a confound — see the no-op argument above.)

| Question | Answer | Status |
| --- | --- | --- |
| Effect of **fsx** alone (RankAlign+fsx − Plain RankAlign), self side | 81.63 − 80.82 = **+0.81** | ✅ |
| Effect of **TC** on top of fsx (no vlo), self side | 86.70 − 81.63 = **+5.07** (paired-bootstrap +4.74) | ✅ |
| Effect of **vlo** on top of fsx + TC-self, self side | 81.31 − 86.70 = **−5.39** | ✅ |
| Effect of **vlo** on top of fsx (no TC) | ? | ❌ need 1 new training run: (fsx=Y, vlo=Y, TC=none) |
| Effect of **vlo** alone (Plain RankAlign+vlo − Plain RankAlign) | ? | ❌ need 1 new training run: (fsx=N, vlo=Y, TC=none) |
| Does **fsx+TC-self beat plain RankAlign**, self side? | 86.70 − 80.82 = **+5.88** | ✅ |
| Does **fsx+vlo+TC-self beat plain RankAlign**, self side? | 81.31 − 80.82 = **+0.49** | ✅ |
| Does **fsx+vlo+TC-neg beat plain RankAlign**, neg side? | 82.01 − ? = ? | ⚠ need to re-eval Plain RankAlign on neg ref |
| Does **TC alone help** (Plain RankAlign+TC − Plain RankAlign), no fsx? | ? | ❌ need 1 new training run: (fsx=N, vlo=N, TC=self) |

### Direction of the fsx+vlo+TC vs fsx+TC comparison

A flagged finding worth highlighting: on the self side, on top of
fsx+TC-self, **adding vallogodds drops gen-ROC by 5.39 points**
(86.70 → 81.31). And on top of plain RankAlign, the kitchen sink
(fsx+vlo+TC-self) gives only **+0.49 points** vs plain RankAlign
(81.31 vs 80.82). So most of TC's benefit (the +5.07 pref-loss-only
delta) comes specifically from the **fsx + TC-self combination
without vlo**, not from the fsx+vlo+TC version. This is consistent
with the "fuller" recipe's −3.51 finding from
[`docs/per_task_bootstrap_analysis.md`](per_task_bootstrap_analysis.md)
once we strip the NLL noise out.

## Headline numbers we can already compute

Gen-ROC × 100, mean(std) across 10 rosch tasks, gemma-2-2b epoch2,
pref-loss-only. From
[`outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md`](../outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md).

| Recipe | TC | fsx | vlo | gen-ROC self | gen-ROC neg |
| --- | --- | --- | --- | --- | --- |
| Plain RankAlign (semi=0.1, no-op) | none | N | N | 80.82 (9.40) | — (not run yet) |
| RankAlign+fsx | none | Y | N | 81.63 (8.14) | 82.96 (9.11) |
| RankAlign+fsx + offline self-TC | tc-self | Y | N | 86.70 (6.94) | — (not run) |
| RankAlign+fsx + offline neg-TC | tc-neg | Y | N | — (not run) | 80.82 (12.87) |
| RankAlign+fsx + offline self-TC + vlo | tc-self | Y | Y | 81.31 (8.21) | — (not run) |
| RankAlign+fsx + offline neg-TC + vlo | tc-neg | Y | Y | — (not run) | 82.01 (11.55) |

(All May-2 rows used `--semi-supervised 0.1` which collapses to plain
preference loss for pref-only weights; the `semi` column is dropped
because it's mathematically inert here.)

Clean isolated deltas from this table:

- **fsx alone (no TC):** 81.63 − 80.82 = **+0.81** (self side). Effectively zero — fsx by itself doesn't move the needle on this transfer task.
- **TC-self on top of fsx (no vlo):** 86.70 − 81.63 = **+5.07** (this is the +4.74 paired-bootstrap effect, exact match modulo rounding).
- **vlo on top of fsx + TC-self:** 81.31 − 86.70 = **−5.39**. Adding vlo *erases* most of TC's benefit when combined with fsx + TC-self.
- **fsx+TC-self vs Plain RankAlign:** 86.70 − 80.82 = **+5.88** (clean — *the* answer to "does fsx+TC beat plain RankAlign": yes, by ~6 points).
- **fsx+vlo+TC-self vs Plain RankAlign:** 81.31 − 80.82 = **+0.49** (the kitchen sink barely beats plain RankAlign — the vlo addition kills most of TC's gain).

## What's still missing — and what would actually close the gap

Two kinds of gaps remain after recognising semi-as-no-op:

### Gap 1: Eval-ref coverage on existing checkpoints (cheap to close)

The May 2 checkpoints are still on disk (under `models/`). We just
never ran them under all 4 eval refs. To finish the **self vs neg**
side-by-side we'd need:

- **Plain RankAlign** (`full-completion_semi0.1`): need `neg` ref (and
  ideally `basetyp`/`basetypneg`). Currently only have `self`.
- **RankAlign+fsx + vlo + TC-self** (`tc-self_..._vallogodds_semi0.1`): need
  `neg` ref (and `basetyp`).
- **RankAlign+fsx + vlo + TC-neg** (`tc-neg_..._vallogodds_semi0.1`): need
  `self` ref (and `basetypneg`).

That's roughly 3 checkpoints × 1–3 missing refs × 10 rosch tasks = **30–90
short eval jobs**. Each is a few minutes on a single GPU, so this is
the cheap way to fully populate the gen-ROC self/neg comparison for
the cells we already have.

### Gap 2: 3 missing training runs

Even after the eval re-runs above, 6 cells in the matrix have no
checkpoint at all:

|   | TC=none | TC=self | TC=neg |
| --- | --- | --- | --- |
| **fsx=N, vlo=N** | ✅ have | ❌ missing — *isolates "TC alone, no fsx"* | ❌ missing |
| **fsx=N, vlo=Y** | ❌ missing — *isolates "vlo alone, no fsx, no TC"* | ❌ missing | ❌ missing |
| **fsx=Y, vlo=Y** | ❌ missing — *isolates "vlo alone, on top of fsx, no TC"* | ✅ have | ✅ have |

The minimum to answer all of the user's questions cleanly is **2–3 new
training runs**:

1. **RankAlign+fsx+vlo (no TC)** = `(fsx=Y, vlo=Y, TC=none)`.
   Pairs with RankAlign+fsx (have) to isolate **+vlo on top of fsx**.
   Pairs with `RankAlign+fsx+vlo+TC-self` (have) to isolate
   **+TC on top of fsx+vlo**.

2. **RankAlign+vlo (no fsx, no TC)** = `(fsx=N, vlo=Y, TC=none)`.
   Pairs with Plain RankAlign (have) to isolate **+vlo alone** — no
   fsx, no TC.

3. (**Optional**) **RankAlign+TC-self (no fsx, no vlo)** = `(fsx=N, vlo=N, TC=self)`.
   Pairs with Plain RankAlign (have) to ask "does TC alone help, no
   fsx?".

Two runs (#1 and #2) is the minimum if we only care about the two
remaining "what does X add" questions (vlo on top of fsx, and vlo
alone). Three (#1+#2+#3) closes the matrix completely on the
left-half (TC ∈ {none, self}). The (TC=neg, fsx=N, *) cells are
still empty after that, but neg-TC is mostly redundant analytically
once we know self-TC's behavior.

The launcher [`scripts/run_train_membership_pref_knob_ablation.sh`](../scripts/run_train_membership_pref_knob_ablation.sh)
covers earlier proposals; it should be **updated** to drop the now-have
cell (Plain RankAlign) and add the (fsx=N, vlo=Y, TC=none) cell instead.

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
