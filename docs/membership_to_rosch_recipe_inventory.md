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

> **Key insight: in g-mode + pref-only training, both `--semi-supervised 0.1`
> AND `--validator-log-odds` are mathematical no-ops.**
>
> **Why `--semi-supervised 0.1` is a no-op.** Looking at
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
> Both branches collapse to `preference_loss`.
>
> **Why `--validator-log-odds` is a no-op (in g-mode + pref-only).** The
> flag is consulted in only two places that touch the gradient:
>
> 1. **Preference-loss scoring** (`_online.py:2785` / `ref.py:2523`):
>    `if train_g_or_d == 'd' and validator_log_odds: <log-odds> else: <log-probs>`.
>    The condition requires d-mode. In g-mode this branch is never
>    taken; `score_i, score_j` are always `sum_completion_logprobs(...)`,
>    independent of vlo.
>
> 2. **Validator NLL loss** (`_online.py:2892` / `ref.py:2600`): vlo
>    switches between BCE-on-log-odds and `-log P(correct)`. But this
>    loss is multiplied by `nll_validator_weight = 0` in pref-only.
>
> Pair selection (the `--delta` filter) uses `logprobs_last_layer` =
> `log P("Yes" | prompt)` in g-mode (`ref.py:1014-1022`,
> `_online.py:1014-1022`) — raw log-prob, vlo not consulted.
>
> All other vlo references are tracking-only (always pass
> `validator_log_odds=True` regardless of flag) or cosmetic (print
> statement, model directory suffix).
>
> **So: a pref-only g-mode May-2 run with `--validator-log-odds
> --semi-supervised 0.1` is mathematically identical to one with both
> flags dropped.** The only externally-visible differences are the
> model directory name and a print line at startup.

### Empirical check: should-be-identical runs differ by ~5 points

If the no-op argument is right, then the May-2 `*_vallogodds_semi0.1`
checkpoints should produce the same gen-ROC as their May-13 (no-vlo,
no-semi) counterparts. They don't:

| Should-be-identical pair | May-13 (no-vlo, no-semi) | May-2 (vlo+semi) | Δ |
| --- | --- | --- | --- |
| fsx + TC-self, self ref | 86.70 | 81.31 | **−5.39** |
| fsx + TC-neg, neg ref | 80.82 | 82.01 | +1.19 |

The May-2 minus May-13 gaps cannot be vlo or semi (both no-ops). They
must be **cohort drift**: different git commit, different
`eval_by_claude.py` version, different RNG seed for pair sampling, or
some other setup variable that changed between May 2 and May 13.

> ⚠ **Implication for cross-cohort comparisons.** Any May-2-minus-May-13
> delta in the headline table below is at least *partly* code/eval drift,
> not the labelled flag effect. The cleanest pref-loss-only deltas we
> have are **within-May-13** cells.

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

Because vlo is a no-op in g-mode + pref-only, the (vlo=Y) row of the
matrix collapses onto the (vlo=N) row mathematically. We keep both
rows below for bookkeeping (different on-disk checkpoints, different
training cohorts) but flag them as **mathematically equivalent**.

|   | TC=none | TC=self | TC=neg |
| --- | --- | --- | --- |
| **fsx=N, vlo=N** | ✅ `full-completion_semi0.1` (May 2) — Plain RankAlign  *eval refs: self only* | ❌ missing | ❌ missing |
| **fsx=Y, vlo=N** | ✅ `full-completion_force-same-x` (May 13)  *eval refs: all 4* | ✅ `tc-self_full-completion_force-same-x` (May 13)  *eval refs: self, basetyp* | ✅ `tc-neg_full-completion_force-same-x` (May 13)  *eval refs: neg, basetypneg* |
| **fsx=N, vlo=Y** | ≡ (fsx=N, vlo=N) — *no-op*, no separate ckpt | ≡ (fsx=N, vlo=N, TC=self) — no separate ckpt | ≡ (fsx=N, vlo=N, TC=neg) — no separate ckpt |
| **fsx=Y, vlo=Y** | ≡ (fsx=Y, vlo=N) — *no-op*, no separate ckpt | ✅ `tc-self_..._vallogodds_semi0.1` (May 2) ≡ (fsx=Y, vlo=N, TC=self) mathematically; in practice differs by 5.4pt cohort drift  *eval refs: self only* | ✅ `tc-neg_..._vallogodds_semi0.1` (May 2) ≡ (fsx=Y, vlo=N, TC=neg) mathematically  *eval refs: neg only* |

Key:

- **✅** = checkpoint trained, evaluated on all 10 rosch tasks at epoch2 (eval refs listed in italics).
- **❌ missing** = no training run for this combination.
- **≡** = mathematically identical to another cell; no new information to gather from a separate run with this flag combo.

After collapsing for the no-op, the **distinct cells we actually need
to fill** are 6 (two TC values × {fsx=N, fsx=Y}, plus one TC=none ×
fsx=N which we already have). With the cells we have, the
**outstanding distinct-cell coverage is 4 / 6**.

The **eval-ref coverage is uneven** across the May 2 cells — they were
each evaluated under a single ref matched to the TC choice, not all 4
refs. The May 2 checkpoints are still on disk under `models/` (verified
2026-05-14), so re-running missing eval refs is cheap (a few minutes
per (task, ref) instead of a full retrain).

## What this inventory says about the user's questions

(Numbers below are gen-ROC × 100, mean across 10 rosch tasks, gemma-2-2b epoch2, evaluated under the matching eval ref. **vlo and semi are no-ops** in g-mode + pref-only — any apparent vlo/semi effect is cohort drift, not the flag.)

| Question | Answer | Status |
| --- | --- | --- |
| Effect of **fsx** alone (RankAlign+fsx − Plain RankAlign), self side | 81.63 − 80.82 = **+0.81** | ⚠ cross-cohort (May 13 vs May 2) — ~5pt drift possible |
| Effect of **TC** on top of fsx (no vlo), self side | 86.70 − 81.63 = **+5.07** (paired-bootstrap +4.74) | ✅ within May 13 — clean |
| Effect of **vlo** on top of fsx + TC-self | **0 by code inspection** (no-op in g-mode + pref-only) | ✅ verified in source |
| Effect of **vlo** alone (Plain RankAlign+vlo − Plain RankAlign) | **0 by code inspection** | ✅ verified in source |
| Does **fsx+TC-self beat plain RankAlign**, self side? | 86.70 − 80.82 = **+5.88** | ⚠ cross-cohort drift confounded with the +0.81 fsx delta above |
| Does **fsx+vlo+TC-self beat plain RankAlign**, self side? | mathematically same as fsx+TC-self vs plain RankAlign | ⚠ same caveat |
| Does **fsx+vlo+TC-neg beat plain RankAlign**, neg side? | mathematically same as fsx+TC-neg vs plain RankAlign | ⚠ need plain RankAlign neg ref |
| Does **TC alone help** (Plain RankAlign+TC-self − Plain RankAlign), no fsx? | ? | ❌ need 1 new training run: (fsx=N, vlo=N, TC=self) |

### What's actually clean and what's drift

After applying the no-op argument:

- **Within-May-13 deltas** are clean: TC-self on top of fsx adds
  **+5.07** (matches the paired-bootstrap +4.74). This is the headline
  finding and the only thing immune to cohort drift.
- **Cross-cohort May-2 vs May-13 deltas** are partly drift. Empirical
  evidence for the drift size: the May-2 fsx+TC-self+vlo+semi run
  (which is *mathematically identical* to May-13 fsx+TC-self) differs
  by **−5.39 points**. So the +0.81 "fsx alone" delta is plausibly in
  the noise of cohort drift; the +5.88 "fsx+TC vs plain RankAlign"
  delta has the same caveat layered on.

To disentangle the +5.07 (clean, TC effect) from the cross-cohort
drift, we'd need to **retrain Plain RankAlign on the May 13 commit**
(no fsx, no TC, no vlo, no semi) and re-derive the comparisons from
that. Not from a vlo+semi run.

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

Deltas from this table, with drift caveats:

- **TC-self on top of fsx (no vlo), within May 13:** 86.70 − 81.63 =
  **+5.07** (the +4.74 paired-bootstrap effect). This is the single
  cleanest delta — same cohort, same commit, same eval pipeline.
- **fsx alone (no TC), cross-cohort:** 81.63 − 80.82 = **+0.81**.
  Plausibly noise — drift between May 2 and May 13 alone is ~5 points
  on should-be-identical runs (see "Empirical check" above). Treat as
  inconclusive until plain RankAlign is retrained on the May 13 commit.
- **vlo on top of fsx+TC-self, cross-cohort:** 81.31 − 86.70 = **−5.39**.
  This number was previously labelled as a vlo effect; **after code
  inspection it can't be**. It's pure cohort drift (likely some mix of
  commit drift, RNG seed drift, and/or eval pipeline drift).
- **fsx+TC-self vs Plain RankAlign, cross-cohort:** 86.70 − 80.82 =
  **+5.88**. Confounds the clean +5.07 within-cohort TC effect with
  cross-cohort drift; expect the "true" delta to be in roughly the
  +0 to +10 range depending on which side the drift falls.
- **vlo on its own, on its own and on top of fsx, on top of fsx+TC:**
  **all 0 by code inspection** (no-op in g-mode + pref-only).

## What's still missing — and what would actually close the gap

Once vlo is recognised as a no-op, the matrix collapses to a 3 × 2
grid in (TC, fsx). We have **4 of those 6 cells**; the missing two
are the actually-informative new training runs.

### Gap 1: Eval-ref coverage on existing checkpoints (cheap to close)

The May 2 checkpoints are still on disk (under `models/`). They have
limited eval-ref coverage. To finish the **self vs neg** side-by-side
on cells we already have:

- **Plain RankAlign** (`full-completion_semi0.1`): need `neg` ref. We
  currently only have `self`. This unlocks "Does fsx+TC-neg beat
  Plain RankAlign on the neg side?"

(The other May-2 checkpoints are vlo+semi versions of cells we
already have cleanly in May 13, so re-running their missing refs
just gives us drift-confounded duplicates — not worth the GPU time.)

That's **10 short eval jobs** (Plain RankAlign × neg ref × 10 rosch
tasks).

### Gap 2: 2 missing training runs (vlo runs are no-ops, drop them)

After the no-op argument, the (vlo=Y) row gives no new information.
The actually-missing training runs are the two TC-with-no-fsx cells:

|   | TC=none | TC=self | TC=neg |
| --- | --- | --- | --- |
| **fsx=N** | ✅ have (Plain RankAlign) | ❌ **missing** — *isolates "TC alone, no fsx"* | ❌ **missing** — *isolates "neg-TC alone, no fsx"* |
| **fsx=Y** | ✅ have | ✅ have | ✅ have |

Plus, for cleanest cross-comparison on the May-13 commit:

- (Optional but worthwhile) **Plain RankAlign retrained on May-13 commit**
  to remove cohort-drift from the existing fsx-vs-no-fsx comparisons.

So the **minimum new training runs** is:

1. **RankAlign+TC-self (no fsx)** = (fsx=N, TC=self). Pairs with Plain
   RankAlign (have) for "does TC alone help?" Pairs with
   RankAlign+fsx+TC-self (have) for "what does fsx add to TC-self?"

2. **RankAlign+TC-neg (no fsx)** = (fsx=N, TC=neg). Symmetric for
   neg-TC; only really needed if we care about the neg side.

3. (Optional) **Plain RankAlign on May-13 commit**: removes the
   cross-cohort drift from the +0.81 / +5.88 deltas above.

The launcher
[`scripts/run_train_membership_pref_knob_ablation.sh`](../scripts/run_train_membership_pref_knob_ablation.sh)
should be **rewritten** to launch (#1, #2, optional #3) — vlo
variants are not worth running.

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
