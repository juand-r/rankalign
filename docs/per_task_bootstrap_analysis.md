# Per-task bootstrap analysis for quick-iter results

## Why this exists

Small rosch categories (~50 positives / 50 negatives each) make per-task
gen-ROC noisy. When we eyeballed the membership-sans-rosch-v0 → rosch
heatmaps and bar plots, individual tasks looked like coin flips even
where the underlying methods were genuinely different. We needed two
things:

1. **Honest uncertainty bars** on each per-task gen-ROC, so we can tell
   when a per-task win is just sample noise vs a real signal.
2. **Pairwise comparisons** between methods (e.g. "is offline self-TC
   reliably better than RankAlign on this task?") that aren't fooled by
   the within-task noise that hits both methods equally.

Both are now built into
[`scripts/plot_membership_to_rosch_per_task.py`](../scripts/plot_membership_to_rosch_per_task.py).

## Code locations

Everything lives in one script:
[`scripts/plot_membership_to_rosch_per_task.py`](../scripts/plot_membership_to_rosch_per_task.py).
Key functions:

| Function | Purpose |
| --- | --- |
| `_auc_from_scores(y, gen)` | Manual AUC via Mann–Whitney U (rank-sum). Fast, handles ties. |
| `bootstrap_auc_ci(scores_path)` | Marginal bootstrap CI on a single (variant, eval-ref, task) cell. |
| `paired_bootstrap_auc_diff(csv_a, csv_b)` | Paired bootstrap CI on AUC<sub>A</sub> − AUC<sub>B</sub> across two variants on the same items. |
| `build_scores_index(scores_dir)` | Walks `outputs-quickiter/` and indexes every `scores_*.csv` by `(variant_id, eval_ref, eval_task)`. |
| `make_plot(...)`, `make_heatmap(...)` | Per-task bar plots (with marginal CIs) and heatmaps. |
| `make_delta_plot(...)`, `make_delta_table(...)` | Per-task paired-Δ plots and markdown summary. |

Bootstrap settings: 1000 reps, 95% percentile CIs, seed 0 for
reproducibility, default score column = `gen_score_typcorr` (matches
`variant == "tc"` in the long-form metrics CSVs). All knobs are module
constants near the top of the file.

## Method 1 — Marginal bootstrap (per-cell CI)

**Question it answers:** for a single (variant, eval-ref, task) cell,
how uncertain is the gen-ROC point estimate?

**Procedure (`bootstrap_auc_ci`):**

1. Load the score CSV. Extract `y` (binary label) and `gen_score_typcorr`.
2. For each of *B* = 1000 reps:
   - Resample positive *item indices* with replacement.
   - Resample negative *item indices* with replacement.
   - Re-compute AUC on the resampled item set.
3. Take the 2.5th / 97.5th percentile of the *B* AUCs.

**Why item-level (not pair-level)?** AUC is computed over the
*n*<sub>pos</sub> × *n*<sub>neg</sub> pairs implicit in the data, but
those pairs are **not** independent — pair (p<sub>1</sub>, n<sub>1</sub>)
and pair (p<sub>1</sub>, n<sub>2</sub>) share an item, so they're
correlated. The independent sampling unit is the **item**, not the pair.
Resampling pairs (or applying a Wilson interval to "fraction of correct
pair orderings") would give CIs that are too tight, because the
within-item correlation gets ignored.

**Where it shows up:**
- [`outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_gen_roc_self.png`](../outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_gen_roc_self.png)
- [`outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_gen_roc_neg.png`](../outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_gen_roc_neg.png)

Each bar in those plots has a 95% marginal CI as an error bar. The CIs
are wide (~10–15 points), which is the honest visual answer to "how
much should I trust per-task differences?": for most pairs of variants,
**the marginals overlap and you can't distinguish them on one task**.

That's *not* the same as saying there's no real difference — see Method
2 below.

## Method 2 — Paired bootstrap (per-task Δ between two variants)

**Question it answers:** is the variant reliably different from
RankAlign **on this specific task**, given that we score both variants
on the same items?

**Why marginal CIs are the wrong tool here.** When two variants are
evaluated on the same task, they share the same items. If a category
happens to contain mostly easy items, both variants get a higher AUC; if
it's mostly hard items, both get lower. The two AUCs move together
across hypothetical resamples of the items. The marginals each include
that within-task item noise, so they're wider than the noise on the
*difference* — that shared component cancels in subtraction.

This is exactly the paired t-test vs unpaired t-test distinction, lifted
to AUC.

**Procedure (`paired_bootstrap_auc_diff`):**

1. Load both score CSVs. Inner-merge them on `(category, member, label)`
   so each merged row has the *same item* with two scores: `gen_a` and
   `gen_b`.
2. Compute AUC<sub>A</sub>, AUC<sub>B</sub>, and the point estimate
   Δ = AUC<sub>A</sub> − AUC<sub>B</sub>.
3. For each of *B* = 1000 reps:
   - Resample positive *item indices* with replacement.
   - Resample negative *item indices* with replacement.
   - Apply **the same** index list to *both* `gen_a` and `gen_b`.
   - Compute AUC<sub>A,b</sub> and AUC<sub>B,b</sub> on the resampled
     rows; record Δ<sub>b</sub> = AUC<sub>A,b</sub> − AUC<sub>B,b</sub>.
4. CI on Δ = 2.5th / 97.5th percentile of {Δ<sub>b</sub>}.

**Significance rule.** A per-task comparison is "significant at 95%"
exactly when the CI on Δ excludes 0 (`lo > 0` or `hi < 0`).

**Where it shows up:**
- [`outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign_self.png`](../outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign_self.png)
- [`outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign_neg.png`](../outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign_neg.png)
- [`outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign.md`](../outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign.md)

Each Δ bar's height is the point estimate (× 100). The error bar is the
paired-bootstrap 95% CI on Δ. **Stars and full opacity** mark Δ whose
CI excludes 0 (a real win or loss). **Faded bars** are non-significant
(can't tell either way). The horizontal line at 0 is the null
hypothesis.

The markdown table dumps every cell as `+5.6 [+0.4, +12.2]` with
**bold** for CIs that exclude 0. That's the right artifact for putting
exact numbers in a paper.

## Comparisons currently rendered

For each side (self-eval and neg-eval), the variant on the right of the
Δ is RankAlign. The variant on the left and the eval-refs used are:

**Self side** (`SELF_DELTA`):

| Variant A | A's eval-ref | RankAlign's eval-ref |
| --- | --- | --- |
| SFT (variant 6) | self | self |
| Offline self-TC (variant 2) | basetyp | self |
| Online self-TC (variant 3) | self | self |

**Neg side** (`NEG_DELTA`):

| Variant A | A's eval-ref | RankAlign's eval-ref |
| --- | --- | --- |
| SFT (variant 6) | neg | neg |
| Offline neg-TC (variant 7) | basetypneg | neg |
| Online neg-TC (variant 8) | neg | neg |

These match the canonical "best per row" eval-ref convention used in
the 4-table layout (see
[`docs/results_table_format.md`](results_table_format.md)). Variant
IDs match the SIG_MAP at the top of the plot script and the layout
in [`scripts/_table_format_4tables.py`](../scripts/_table_format_4tables.py).

## Caveat: eval-ref mismatch in the offline-TC comparison

When the comparison is `offline TC (under basetyp[neg]) vs RankAlign
(under self/neg)`, the two AUCs use **different scoring rules** —
different typicality references subtracted from the generator score.
The paired bootstrap merges on items, not on scoring rule, so the Δ
should be read as

> "the AUC offline-TC achieves under its canonical operating point,
> minus the AUC RankAlign achieves under its canonical operating point,
> on the same items."

That's the comparison the 4-table layout already endorses, and is
probably what we'd report in a paper. But if you want to answer "does
TC-trained generation rank items better than RankAlign-trained, holding
the eval rule fixed?", you'd flip the eval-refs in `SELF_DELTA` /
`NEG_DELTA` to a matched pair (e.g. both under `self`, or both under
`basetyp`) and rerun.

For online TC and SFT the eval-refs already match RankAlign's, so those
rows don't have this caveat.

## Findings (membership-sans-rosch-v0 → rosch, gemma-2-2b epoch2)

Counted from
[`per_task_delta_vs_rankalign.md`](../outputs-quickiter/membership-sans-rosch-v0-to-rosch/per_task_delta_vs_rankalign.md).
Three-way classification (sig wins / sig losses / non-significant; sig
= 95% paired-bootstrap CI on Δ excludes 0):

| Comparison vs RankAlign | sig win | sig loss | non-sig |
| --- | --- | --- | --- |
| **Self side** | | | |
| SFT (self eval-ref) | 2/10 | 0/10 | 8/10 |
| Offline self-TC (basetyp eval-ref) | **6/10** | **0/10** | 4/10 |
| Online self-TC (self eval-ref) | 3/10 | 0/10 | 7/10 |
| **Neg side** | | | |
| SFT (neg eval-ref) | 0/10 | **7/10** | 3/10 |
| Offline neg-TC (basetypneg eval-ref) | 2/10 | **6/10** | 2/10 |
| Online neg-TC (neg eval-ref) | 3/10 | 2/10 | 5/10 |

### Headline takeaways

1. **Offline self-TC is the clear winner against RankAlign on the self
   side**: 6 sig wins, 0 sig losses, point estimate Δ > 0 in *all 10*
   tasks (the 4 non-significant ones are still +0.8, +1.5, +2.7, +2.7).
2. **The benefit of TC is asymmetric in eval direction.** The same
   training recipe that wins on the self side (offline TC) actually
   *loses* to RankAlign on most of the neg-side tasks (6 sig losses).
   Online neg-TC has a slightly better picture (3 wins, 2 losses) but
   it's still much messier than the self side.
3. **SFT on the neg side is genuinely bad** (7 sig losses out of 10 vs
   RankAlign), most extremely on rosch-vehicle (Δ ≈ −30). Some of this
   may be a metric artifact (SFT's neg-prompt distribution might be a
   poor normalizer post-finetune), some may be real degradation.

### Caveat: this is the *isolated* TC effect, not "TC's effect in any recipe"

The May 13 recipe used here is a **minimal-controlled comparison**: RankAlign
with the preference loss alone, then the same baseline with `+ tc-self`
toggled on. Everything else (`nllv1.0` on validator, `nllg1.0` on
generator, `vallogodds`, `semi0.1`) was removed from earlier May 2 / May 4
recipes precisely so the *only* thing varying between RankAlign and
offline-self-TC is the TC training objective.

Older membership-sans-rosch-v0 → rosch evals (in `outputs/`, May 1–4)
trained gemma-2-2b under a fuller recipe
(`full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` and
`tc-self_full-completion_*_vallogodds_semi0.1`). Aggregating those
across the 10 rosch tasks (see
[`outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md`](../outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md))
gives:

| Recipe family (gemma-2-2b, epoch2) | no-TC | offline self-TC | Δ |
| --- | --- | --- | --- |
| Minimal (May 13) | 81.63 | 86.37 | **+4.74** |
| Full (May 2): with NLL matched | 84.82 | 85.14 | +0.32 |
| Full (May 2): no NLL on TC side | 84.82 | 81.31 | −3.51 |

The minimal-recipe number is the cleanest measurement of "what does adding
TC give us on top of RankAlign?" It's not a *better* number, it's a
*more interpretable* one — there are no other knobs varying. The
shrinkage in the full recipe is consistent with NLL + semi-supervision
already supplying some of the calibration that TC was correcting for.
A controlled ablation that turns those auxiliaries on/off one at a time
would be the right tool to pin down which one is competing with TC.
Reporting the +4.7 number in a paper should come with this caveat.

### What "asymmetric in eval direction" might be

Three competing explanations, none of which is yet ruled out:

- **The neg prompt is noisier** ("Do you think a robin is NOT a bird?"
  is less natural; *P*<sub>θ</sub>(*y* | neg(*x*)) is harder to
  estimate cleanly).
- **TC training interacts with the neg-prompt distribution** in ways
  that flatten it, making the neg-eval normalizer less informative
  exactly for TC-trained models.
- **Pure metric story**: switching the eval-ref between models that
  were/weren't trained with TC changes the comparison, not the
  underlying generator quality.

Disentangling these would mean re-running the deltas under a
*matched-eval-ref* setup (e.g. all models scored under `basetyp`) and
seeing whether the asymmetry survives.

## How to regenerate

```bash
source /u/jdr/venvs/venv_lexcons/bin/activate
python scripts/plot_membership_to_rosch_per_task.py
```

End-to-end runtime is ~50 s on a single CPU (262 score CSVs × 1000
bootstrap reps; the paired-bootstrap loop dominates). Outputs are
written to
`outputs-quickiter/membership-sans-rosch-v0-to-rosch/`:

- `per_task_gen_roc_self.png`, `per_task_gen_roc_neg.png` — bar plots
  with marginal 95% CIs.
- `per_task_gen_roc_heatmap_self.png`, `per_task_gen_roc_heatmap_neg.png`
  — 5-column heatmaps.
- `per_task_gen_roc_table.md` — wide HTML table (4+4 layout).
- `per_task_delta_vs_rankalign_{self,neg}.png` — paired-Δ bar plots.
- `per_task_delta_vs_rankalign.md` — paired-Δ markdown table.

## Extending to other (training-task, model) pairs

The script is currently hardwired to
`membership-sans-rosch-v0` × `gemma-2-2b` × epoch 2 via the
`MODEL`, `TRAIN_TASK`, `EPOCH`, and `OUT_DIR` constants near the top
plus the `parse_filename` regex. To run the same analysis on
- rosch-fb → rosch OOD tasks
- AmbigQA delta sweep
- 2b-it variants

the simplest path is to fork the file (e.g.
`plot_rosch_fb_per_task.py`) and update those four constants plus the
`TASKS_BY_OVERLAP` ordering. The bootstrap helpers themselves are
task-agnostic and can be lifted to a shared module if we end up with
3+ copies.
