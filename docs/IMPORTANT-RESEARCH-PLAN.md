# IMPORTANT — Research Plan

This document is the canonical plan for "why aren't our novelties beating
RankAlign, and what do we do about it?" It is meant to be appended to as
we learn. Everything below is the agreed setup; new findings go at the
bottom of the relevant section, dated.

---

## 1. The big picture

**Goal.** This year we made three changes on top of last year's RankAlign
baseline. Empirically they **don't help over RankAlign**. We want to
understand why — especially TC, which is the most surprising one.

### Three contributions, in priority order

1. **TC during training** (priority 1, most surprising it doesn't help).
   Two flavors: **self-TC** and **neg-TC** (neg-TC was theoretically
   derived; expected to help in some regimes).
2. **`force-same-x`** during training (priority 2). We suspected it
   would help — pairs would compare different completions for the same
   input rather than mixing inputs. It didn't help much; itself a
   mystery.
3. **`comb` loss** (priority 3 — "who cares"). Combines preference +
   NLL on the 10% labeled portion (and pref-only on the 90% unlabeled).
   Mainly added to avoid likelihood collapse / displacement. Gives
   better validator-accuracy numbers but isn't otherwise exciting. We
   suspect it makes essentially no difference because it only kicks in
   on 10% of the data.

### Methodology fix that affects everything

We train TC in the **offline setting**, i.e., the typicality reference
is the *fixed base model*, not the model being fine-tuned. So at eval
time we should be using `--base-typicality --base-model <base>` (e.g.
`google/gemma-2-2b`), not `--self-typicality` on the fine-tune. Using
self-TC at eval breaks the offline match between train-time TC and
eval-time TC, and is probably contributing to the "TC doesn't help"
observation.

Going forward:

- We **always** test with either `--self-typicality` or `--neg-typicality`
  in `eval_by_claude.py`. That single eval run produces a `scores_*.csv`
  with all four columns we care about: `raw`, `tc` (or neg-tc),
  `lenorm`, and `tc + lenorm`. We ALSO always test with --base-typicality.
- Models trained with **self-TC** get **self-TC** at eval; models
  trained with **neg-TC** get **neg-TC** at eval. (Match train and eval
  TC type.)
- Going forward we **also** sweep with `--base-typicality` so the eval
  reference is the same fixed base used during training. Initial
  experiments showed `basetyp-` / `basetypneg-` improved results
  slightly but **not enough to flip the rankings between methods**.
- We are **not** going to re-run the existing experiments from scratch
  to fix this. It isn't a "bug" so much as we weren't testing the
  exactly-right thing. From now on, when we launch new runs, we do it
  right.

### Known parameter-cleanup items (do for new runs, don't re-run old)

- Settings #5 and #8 (see table below) currently have
  `--validator-log-odds` set during train. They shouldn't — `vlo` is
  conceptually the "right way to do `comb`" and these two settings
  drop `comb`. Remove `--validator-log-odds` from new launches of #5
  and #8.

---

## 2. The 12 training settings (gemma-2-9b-it humaneval reference; #10–12 added for gemma-4 run)

These are the settings encoded in
[`scripts/run_train_humaneval.sh`](../scripts/run_train_humaneval.sh) and
[`scripts/run_train_humaneval_neg.sh`](../scripts/run_train_humaneval_neg.sh).
The same naming scheme generalizes to other tasks (rosch / membership,
ambigqa, plausibleqa, ifeval, etc.).

| # | Setting | Loss | Semi/lo | log-odds | force-same-x | TC | Purpose |
|---|---|---|---|---|---|---|---|
| 1 | **SFT-lo** | sft | labelonly | — | — | — | NLL-only baseline. SFT on the 10% labeled subset only. |
| 2 | **RankAlign** | pref-only | semi | — | — | — | Last year's baseline. With `--semi-supervised 0.1` and pref-only loss this is equivalent to non-semi: pref applies to both labeled and unlabeled portions. |
| 3 | **New+fsx** | comb | semi | ✓ | ✓ | — | All non-TC novelties together. comb-loss + fsx + vlo, no TC. (vlo is the right way to do comb.) |
| 4 | **New+fsx+tc** | comb | semi | ✓ | ✓ | self | All three contributions together: comb + fsx + self-TC + vlo. |
| 5 | **RankAlign+fsx+tc** | pref-only | semi | ✓ (bug) | ✓ | self | Like #4 but drops `comb`. Lets us check whether #4's outcome is actually due to comb or to TC. **Bug:** has `--validator-log-odds` during train; it shouldn't, vlo belongs with comb. Remove for new runs. |
| 6 | **RankAlign+tc** | pref-only | semi | — | — | self | RankAlign + self-TC alone. **The cleanest "does TC alone help?" probe.** |
| 7 | **New+fsx+negtc** | comb | semi | ✓ | ✓ | neg | Like #4, neg-TC instead of self-TC. |
| 8 | **RankAlign+fsx+negtc** | pref-only | semi | ✓ (bug) | ✓ | neg | Like #5, neg-TC. Same `--validator-log-odds` bug. |
| 9 | **RankAlign+negtc** | pref-only | semi | — | — | neg | RankAlign + neg-TC alone. **The cleanest "does neg-TC alone help?" probe.** |
| 10 | **RankAlign+fsx** | pref-only | semi | — | ✓ | — | RankAlign + fsx alone (no TC, no comb, no vlo). **The cleanest "does fsx alone help?" probe.** Added in the gemma-4 humaneval-v2.1correct-multi run. |
| 11 | **New+tc** | comb | semi | ✓ | — | self | comb + vlo + self-TC, no fsx. Isolates TC on top of the new loss without fsx confound. Added in the gemma-4 humaneval-v2.1correct-multi run. |
| 12 | **New+negtc** | comb | semi | ✓ | — | neg | comb + vlo + neg-TC, no fsx. Isolates neg-TC on top of the new loss without fsx confound. Added in the gemma-4 humaneval-v2.1correct-multi run. |

So the diagnostic shape is:

- **Does TC alone help?** → #6 / #9 vs #2 (and vs #1).
- **Does TC help on top of new-loss + fsx?** → #3 vs #4 (self) and #3
  vs #7 (neg).
- **Does TC help on top of new-loss alone (no fsx)?** → #3-without-fsx vs #11 (self) and #12 (neg). More directly: #11 vs #6 (adds comb to self-TC) and #12 vs #9 (adds comb to neg-TC).
- **Does fsx contribute given new-loss + TC?** → #4 vs #11 (self), #7 vs #12 (neg).
- **Does fsx contribute given TC alone?** → #6 vs #5, #9 vs #8.
- **Does fsx alone help?** → #10 vs #2.

---

## 3. Typicality-correction focus (priority-1 question)

This is the section we are actively working on. **Everything we do
right now should serve the question: why doesn't training-time TC beat
RankAlign?**

### Headline question

> Why does TC-during-train (especially with corrected
> `base-typicality` eval) not beat RankAlign?

### Follow-up questions

- Are there settings — properties of the data, of the model, of the
  task — where train-time TC **does** beat RankAlign?
- Is test-time TC doing essentially all of the work, leaving nothing
  for train-time TC to add?

### What "beating RankAlign" has to mean

Applying TC or neg-TC **at eval time** nearly always improves
**Pearson correlation** and **gen-ROC** (humaneval is the one
exception). These are the metrics we care about most.

So the bar for #6 / #9 is:

> #6 / #9 must beat #1 / #2 **when all four are evaluated with the
> matching TC variant** (i.e. all four read off the `tc` or `tc+lenorm`
> column of their scores files, with `--base-typicality` against the
> matching base model).

If TC-during-train only helps when comparing **raw** eval columns and
the gap closes once we also TC-correct everyone at eval, that's not
interesting — test-time TC is doing the work. We need it to win on
the TC'd column.

### Comparison set we will focus on

| # | Setting | Eval recipe |
|---|---|---|
| 1 | SFT-lo | `--neg-typicality` and `--self-typicality`, both with `--base-typicality --base-model <base>` |
| 2 | RankAlign | same |
| 6 | RankAlign+tc | `--self-typicality --base-typicality --base-model <base>` |
| 9 | RankAlign+negtc | `--neg-typicality --base-typicality --base-model <base>` |

Hopefully there aren't weird interactions when we combine TC with
fsx and the new loss. We'll come back to those after we have the
TC-alone story straight.

### Three avenues of investigation

We need all three; they are not substitutes.

#### (a) Theoretical re-evaluation

Is there something wrong with the method as proposed? Specific
sub-questions to chase:

- **[Resolved May 10 — see findings subsection below]** Does
  train-time TC inject signal into the gradient *direction* or only
  reweight pairs?
  In the current code: only reweights. The TC offset `−(t_+ − t_-)`
  is a θ-independent constant per pair, so it enters the scalar
  weight `1 − σ(Δ_θ)` but never the direction
  `∇_θ[g_θ(y_+) − g_θ(y_-)]`. This holds for self-TC, neg-TC, and
  the (de facto) base-TC equally — all four cells of the
  self/neg × online/offline 2×2 collapse to "offline" because `t_i`
  is precomputed once with `torch.no_grad()` and stored as a Python
  `float`. **Independent of `--force-same-x`**: fsx changes which
  pairs exist, not whether TC participates in the gradient.
- **Open: does pure reweighting move the corrected eval metric?**
  Per-pair reweighting on a fixed underlying objective generally
  produces modest deltas vs. uniform weighting — which fits the
  empirical observation that #6 / #9 don't beat #2. But we should
  be precise about *when* reweighting can/can't matter (small data?
  early-stopped training? heavy class imbalance in `(t_+ − t_-)`?)
  before concluding pure reweighting is the whole story.
- **Open: does the offline-TC objective `log P_θ(y|x) − log P_base(y)`
  give the right pair-ranking signal in expectation,** even granting
  the gradient-direction issue? Treated as a *population* objective
  rather than a gradient signal, is the maximizer of the corrected
  score on labeled pairs the right thing to want? In particular,
  does any calibration property of `P_base` need to hold for this
  to be a sensible target?
- **Open: under the proposed live-with-grads `t_θ` (Option D in the
  May 10 subsection), characterize the unregularized fixed points.**
  The structural concern is that the gradient rewards making the
  null-context distribution `P_θ(·∣null)` weird/sparse to inflate
  corrected scores cheaply. Look for known phenomena from the
  PMI-based RL literature or the GPT-2 typicality literature that
  give a closed-form characterization of this degeneracy and the
  regularizers that fix it.
- **Open: re-derive neg-TC under live-with-grads end to end.** Both
  `P_θ(y∣x)` and `P_θ(y∣neg(x))` flow through the same θ. Is there
  a self-distillation-like effect, or a sign convention to be
  careful about, that could explain why neg-TC sometimes behaves
  differently from self-TC empirically?
- Sanity-check the derivations of self-TC and neg-TC objectives end
  to end against the code.

##### (May 10) Findings and concrete proposal

**Verification of current behavior** (read of `scripts/ranking_loss_ref.py`):

- `compute_self_typicality_training` and `compute_neg_typicality_training`
  both run **once before the training loop**, under `torch.no_grad()`, and
  store each `t_i` as a Python `float` (lines 200–216, 219–246).
- The training step does `score_i_gen = score_i_gen - typicality_i`
  (line 2388) using those cached scalars. The code comment "online,
  during training" applies only to the subtraction, not to the
  evaluation of `t_i`.
- We do **not** load a separate frozen base model alongside the
  trainable one for typicality. The `--with_ref` flag exists but is
  unused in `run_train_humaneval.sh` / `run_train_semi.sh`, and even
  where it would be active, it does not feed the typicality path
  (line 2395 raises `ValueError("Do LATER")`).
- Cached `t_i` ≈ `log P_base(y_i)` because LoRA adapters are
  zero-initialized at snapshot time (or, for non-LoRA runs, the
  snapshot precedes any optimizer step).

**The 2×2 axes** (replacing the earlier "three cases" framing):

| | Online (current θ, grads on) | Offline (frozen reference, grads zero) |
|---|---|---|
| **self** — subtract `log P(y)` | `log P_θ(y)` — gradient-flowing | `log P_base(y)` |
| **neg** — subtract `log P(y∣neg(x))` | `log P_θ(y∣neg(x))` — gradient-flowing | `log P_base(y∣neg(x))` |

We currently sit in the **offline** column regardless of which
`--self-typicality` / `--neg-typicality` flag is set, because of the
precompute-and-cache implementation.

**Headline gradient result.** Define

$$
\Delta_\theta \;=\; [g_\theta(y_+\mid x) - g_\theta(y_-\mid x)] \;-\; (t_+ - t_-),
$$

with `t_±` constants in θ. Then

$$
\nabla_\theta \log \sigma(\Delta_\theta) \;=\; \bigl(1 - \sigma(\Delta_\theta)\bigr)\cdot \nabla_\theta \bigl[g_\theta(y_+\mid x) - g_\theta(y_-\mid x)\bigr].
$$

The `−(t_+ − t_-)` shift enters only the **scalar weight**
`1−σ(Δ_θ)`. Per pair, TC rescales the gradient magnitude but does
not change its direction. Aggregated across a batch, TC reweights
the mixture of per-pair direction vectors `{v_p}`, but cannot point
the model in directions outside their span. **So train-time TC in
the current code is purely a per-pair sample reweighter, independent
of `--force-same-x`.**

**Why this likely explains the headline observation:**

- The model is never trained to lower `log P_θ(y_+)` under the null
  context — the lever TC is intuitively supposed to pull is absent
  from the gradient.
- Reweighting alone produces only modest changes in final parameters
  vs. RankAlign.
- At eval, the corrected score `g_θ(y_+|x) − log P_base(y_+)` is
  built from a `g_θ` shaped no differently than RankAlign's, just
  with different per-pair weights — so corrected scores end up
  similar.

**Concrete proposal: live-with-grads typicality.**

1. **Drop the precompute.** At each training step, compute
   `t_i = log P_θ(y_i)` (or `log P_θ(y_i ∣ neg(x_i))`) live, against
   the current trainable model, **with grads enabled**, and pass it
   into the same `score_i_gen − typicality_i` subtraction.
2. The gradient then picks up
   `−∂[log P_θ(y_+) − log P_θ(y_-)]/∂θ`, which actively pushes the
   model to lower `log P_θ(y_+)` under the null context (and raise
   `log P_θ(y_-)`) — the structural lever TC was meant to pull.
3. **Add a regularizer to prevent degeneracy.** Without one, the
   gradient rewards making `P_θ(·∣null)` weird/sparse so the
   corrected score can be cheaply inflated. Options: a KL term
   `KL(P_θ(·∣null) ‖ P_base(·∣null))` or a per-completion penalty
   `‖log P_θ(y) − log P_base(y)‖` on training completions.

**Eval consistency under this proposal.**

- The directly-matched eval metric is `--self-typicality` (or
  `--neg-typicality`) evaluated under the **trained** model — not
  `--base-typicality`.
- We should still **also** report `--base-typicality` for two
  reasons: (i) cross-model comparability (`log P_base` is a fixed
  ruler), and (ii) degeneracy diagnostic: if `--self-typicality`
  looks great while `--base-typicality` collapses, that's evidence
  the model has gamed the null-context distribution and the
  regularizer needs to be stronger.

#### (b) Code evaluation — is there a bug?

- Re-read [`scripts/ranking_loss_ref.py`](../scripts/ranking_loss_ref.py)
  paths for `--self-typicality`, `--neg-typicality`, and
  `--typicality-correction`, with an eye to: is TC actually applied to
  the **online** generator scores in the gradient, or only to
  pair-selection scores at the start of training? (See
  [`docs/typicality_and_val_boost_plan.md`](typicality_and_val_boost_plan.md);
  the original concern was that TC was only affecting pair selection,
  not the loss.) Confirm fixed in current code.
- Confirm no silent dtype / sign errors: are we subtracting
  `log P_base(y)` (correct) or adding it? Are the per-token sums lined
  up?
- Confirm the pair-construction pipeline isn't undoing the TC by
  re-normalizing or re-ranking based on raw logprobs.
- Confirm checkpoint loading at eval-time uses the merged model
  (LoRA→merged) so that `--base-typicality --base-model <base>` is
  actually contrasting the fine-tuned weights against the right base.

#### (c) Empirical evaluation — does it work in clean settings?

Two probes, in increasing order of "blame the data, not the method":

1. **Train ≈ test (sanity / generalization probe).**
   Train on the test set (or evaluate on the train set) for a given
   task. If TC-during-train **still** doesn't help in this in-domain,
   memorize-friendly setting, the issue is fundamental, not a
   generalization problem. If it does help here but not on held-out
   test, the issue is specifically generalization.

2. **Toy datasets where TC is *expected* to help.**
   Construct small datasets with this structure:
   - **Correct examples**: high validator score AND low base-model
     `log P(y)` — i.e. the right answer is also a typicality-rare
     completion.
   - **Incorrect examples**: low validator score AND high base-model
     `log P(y)` — i.e. the wrong answer is a typicality-common
     completion.

   In this regime, TC should boost correct examples *more* than
   incorrect examples (it's exactly the bias TC is meant to remove),
   so train-time TC should produce a cleaner training signal than
   raw-pref. If TC-during-train **still doesn't beat RankAlign here**,
   we know the failure isn't about the data — it's about the method or
   the implementation.

   Mirror-image dataset to check the other direction: correct
   examples = high val + high base `log P`, incorrect = low val + low
   base `log P`. TC should *hurt* here, providing a useful negative
   control.

### Concrete next-step checklist (TC focus)

These are the things we should do, in order, to make headway on the
typicality question.

1. Make sure existing #1 / #2 / #6 / #9 runs for at least one task
   (rosch is fully populated; consider it the canonical first task)
   are evaluated with `--base-typicality --base-model <base>` plus
   matching `--self-typicality` or `--neg-typicality`. Compare on the
   `tc` and `tc+lenorm` columns.
2. Run avenue (a): write down the in-pair cancellation argument
   formally. Either it predicts our observation, or it doesn't.
3. Run avenue (b): code audit of the TC paths in
   `ranking_loss_ref.py`. Confirm TC affects the online loss.
4. Run avenue (c.1): in-domain (train ≈ test) sanity probe on a
   single small task.
5. Run avenue (c.2): build the toy "TC should help" dataset. Train
   #2 vs #6 vs #9 on it. This is the critical empirical test.
6. Decide based on results whether to keep TC, modify it, or shelve it.

If we make a dent in the TC question we move on to the
`force-same-x` question (priority 2).

---

## 4. Force-same-x (priority-2 question — paused)

> Why doesn't `--force-same-x` help much, when intuitively pairs that
> compare completions for the *same* input should give a cleaner
> contrastive signal than pairs mixing inputs?

We are deferring deep investigation of this until we've made progress
on TC. Note: per section 3(a), TC and `--force-same-x` may interact in
a way that suppresses the TC gradient inside pairs. Whatever we learn
about TC will likely refine the right question to ask about fsx.

---

## 5. The new (`comb`) loss (priority-3 — back-burner)

`comb` adds NLL on the 10% labeled portion (val-NLL and gen-NLL) on top
of the preference loss. The 90% unlabeled portion gets pref-only.

Hypothesis: because it only kicks in on 10% of the data, `comb` doesn't
materially change much except mild gains in **validator accuracy**
(which we attribute to less likelihood collapse / displacement). We
will revisit only if the TC and fsx investigations point to comb as a
confound.

---

## 6. Findings log (append below as we learn)

### 2026-05-10 — Train-time TC is a per-pair reweighter, not a gradient-direction injector

Verified by code-read of `scripts/ranking_loss_ref.py` and gradient
derivation. Full writeup is in §3(a) under "(May 10) Findings and
concrete proposal." Short version:

- All four cells of the 2×2 (self/neg × online/offline) currently
  collapse to **offline** in our code, because `t_i` is precomputed
  once before training under `torch.no_grad()` and stored as a
  Python `float`. So `t_i` is θ-independent and contributes zero to
  the gradient direction.
- Pair gradient: `∇_θ log σ(Δ_θ) = (1−σ(Δ_θ)) · ∇_θ[g_θ(y_+) − g_θ(y_-)]`.
  The TC offset `−(t_+ − t_-)` only enters the scalar weight, not
  the direction vector. Independent of `--force-same-x`.
- Likely explanation for "TC-during-train doesn't beat RankAlign":
  the model is never trained to lower `log P_θ(y_+)` under the null
  context. Reweighting alone yields only modest deltas vs. RankAlign,
  so eval-time corrected scores look similar.
- Proposed fix: live-with-grads typicality (compute `t_i` against
  current θ at each step, grads enabled), plus a regularizer to
  base prior to prevent null-context degeneracy. Eval reports both
  `--self-typicality` (matched objective) and `--base-typicality`
  (cross-comparable + degeneracy diagnostic).
- The earlier §3(a) bullets that asked whether TC cancels inside
  pairs sharing `x` (and whether self-TC / neg-TC behave
  differently in the gradient) have been rewritten to reflect this
  resolution: the cancellation in the gradient direction is total
  and is not specific to `--force-same-x`; self/neg/base all
  collapse to offline in the current code.

---

## 7. Known issues / TODOs (parked for later)

These were surfaced during the May 10 code-review of
`scripts/ranking_loss_ref.py`. None of them block the current TC
investigation; revisit when we run out of higher-priority work.

### Issue #1 — `pair_is_labeled` drops mixed pairs in non-fsx semi-supervised

**STATUS (2026-05-22): RESOLVED in `scripts/ranking_loss_ref_fix.py`
(g-mode only).** Per-item gating via `is_labeled_i_t` / `is_labeled_j_t`
masks replaces the AND-gate in both val-NLL and gen-NLL terms. The
`pair_is_labeled` AND-gate variable still exists in fix1's
`PairwiseDataset.__getitem__` and gets shipped in the batch dict, but
NO active loss path reads it; in-code `# NOTE/TRAP` comments at both
spots (the `__getitem__` definition and the train-loop pickup) warn
against ever wiring it back into a loss multiplier. Parent
`scripts/ranking_loss_ref.py` is unchanged and still has the bug;
backporting is parked (see TODO list). The original analysis below is
preserved for reference / for the parent-code backport when we get to
it.

**Where (in parent / unfixed).** `scripts/ranking_loss_ref.py` line 2161:
`pair_is_labeled = 1.0 if (is_labeled_i and is_labeled_j) else 0.0`,
then used to gate `nll_validator_loss` and `nll_generator_loss` in the
semi-supervised branch (lines 2615, 2620, 2632).

**Effect.**
- Labeled split is **per prompt** (`split_prompts_labeled_unlabeled`,
  line 477), so all items sharing a prompt are labeled or unlabeled
  together.
- **With `--force-same-x`**: pairs are always within one prompt, so
  `is_labeled_i ≡ is_labeled_j`. Fraction of labeled pairs ≈ ratio
  (e.g. 10% with `--semi-supervised 0.1`). **Not biting.**
- **Without `--force-same-x`**: pairs mix labeled/unlabeled items.
  With ratio 0.1: P(both labeled) ≈ 1%, P(mixed) ≈ 18%, P(both
  unlabeled) ≈ 81%. **The 18% mixed pairs silently get treated as
  unlabeled**, dropping NLL signal that should fire on the labeled
  side.

**Status (updated 2026-05-20).** No on-disk trained model currently
combines `comb` + `--no-force-same-x` + `--semi-supervised` (verified
across `models/`, `models-quickiter/`, `outputs/`, and
`outputs_gemma4_*`), so this bug is **not corrupting any existing
result** as of 2026-05-20. However, the bug **will bite settings #11
and #12** as defined in §2 (comb + vlo + tc-self/tc-neg, no fsx, semi)
the moment we train those — they were added to the plan on 2026-05-19
specifically for the gemma-4 humaneval-v2.1correct-multi run.

**Settings to (re-)train and (re-)eval after the fix lands:**

| # | Setting | Why affected |
|---|---|---|
| 11 | New + PMI [−fsx] | `comb semi + no-fsx` → mixed pairs drop labeled-end NLL |
| 12 | New + NegTC [−fsx] | same as #11 |

Settings **not** affected (no change needed):

- #1 SFT-lo: `labelonly` mode discards unlabeled items entirely
  (`scripts/ranking_loss_ref.py:1574-1576`), so every pair is
  pure-labeled; the gate evaluates to 1 everywhere.
- #2 RankAlign, #5, #6, #8, #9, #10: pref-only weights → NLL terms are
  multiplied by 0, the gate is irrelevant.
- #3, #4, #7: `comb + semi` but **with** fsx → pairs are within one
  prompt, `is_labeled_i ≡ is_labeled_j` by per-prompt split, gate
  evaluates correctly. (Quoted as "Not biting" above.)

Any **future** `sft semi + no-fsx` or `comb semi + no-fsx` run that we
add to the plan beyond #11/#12 should be included in the re-train list.

**Proposed fix when we want it.** Per-end masking instead of
pair-level masking:

```python
nll_validator_loss = -(is_labeled_i * score_correct_i +
                       is_labeled_j * score_correct_j).mean() / 2
nll_generator_loss = -(is_labeled_i * score_gen_i * indicator_i +
                       is_labeled_j * score_gen_j * indicator_j).mean() / 2
```

and gate the labeled-vs-unlabeled split per-end too (or just drop the
`pair_is_labeled` gate in the outer combination).

**Verification recipe before claiming the bug is fixed.** Run a tiny
`comb + semi 0.1 + no-fsx` smoke training (e.g. 50 steps on rosch) on
the fixed code and confirm:
1. `train/nll_validator_loss` and `train/nll_generator_loss` are
   nonzero on **mixed** pairs (currently they are zero).
2. The per-step effective NLL-touched-item count matches the labeled
   ratio (≈ 10% of items per pair, not ≈ 1% of pairs).

### TODO — `--ground-truth-not-validator` flag in `ranking_loss_ref_online.py`

Currently `Z` is sorted by `logprobs_last_layer` (the validator score),
so the pair "winner" is whichever side the validator preferred. In
`force-same-x` mode this is confounded with the validator's biases /
errors: the model can never learn to disagree with the initial
validator inside a labeled prompt group, even when the ground-truth
label says it should.

Add `--ground-truth-not-validator`: when set, pair winner is determined
by the ground-truth label (`get_label`) instead of the validator
score. Use this as a diagnostic: if online-TC + GT-pair-winner beats
RankAlign but online-TC + validator-pair-winner does not, then the
issue is the validator-bias-in-pair-selection loop, not TC itself.

### TODO — KL / penalty regularizer for online TC

Per §3(a) "live-with-grads typicality" proposal: once
`--online-typicality` is enabled, the gradient pushes
`P_θ(y_+ | null) ↓` and `P_θ(y_- | null) ↑` (for self-TC). Without a
prior, this can degenerate — the model can drive its null-context
distribution arbitrarily far from the base, hurting general
language-modeling quality and possibly making the typicality term
itself uninformative.

Add a regularizer term to `ranking_loss_ref_online.py` along the
lines of:

```
L_reg = β * KL( P_θ(· | null)  ||  P_base(· | null) )
```

or a cheaper proxy like `β * |g_θ(y; null) − g_base(y; null)|²`
averaged over the items in the batch. We'll need `P_base` available
during training (load the frozen base model alongside the trainable
one — same setup as `WITH_REF` skeleton already in the code).

Wait until we have at least one training run with `--online-typicality`
that we can compare against the offline / RankAlign baselines on
`rosch-furniture-and-bird` before tuning β.
