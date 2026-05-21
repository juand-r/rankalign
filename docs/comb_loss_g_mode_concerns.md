# `comb` loss in `g` mode — bug-shaped concerns surfaced from persona-v1 results

Written 2026-05-21 from inspection of `scripts/ranking_loss_ref.py` after
the persona-v1 GenROC tables (3 bases × 9 variants) showed the "full new
method" (#4 = comb + fsx + self-TC + vlo, and #7 = neg-TC analogue)
**consistently underperforming** the preference-only siblings #5/#6 (self)
and #8/#9 (neg) across all three base models, both ID and OOD splits.
See [`metrics-from-scores/persona_v1_*.csv`](../metrics-from-scores/) and
[chat 2026-05-21](../) for the headline numbers.

This doc collects the three concrete bug-shaped concerns I'd want to
verify before reading more into the persona-v1 method ranking.

> **Scope.** All persona-v1 training uses `--train_g_or_d g` (g mode).
> Everything below is g-mode-only. The mirror analysis for `d` mode and
> `both` mode would look different and is out of scope here.

---

## TL;DR ranked concerns

| # | Concern | Severity | Affects |
|---|---------|----------|---------|
| 1 | Validator NLL is computed at the wrong position in the wrong sequence | **HIGH** | #3, #4, #7 (every `comb` variant) |
| 2 | `--force-same-x` is mathematically a no-op for persona-v1 | **HIGH** | #3, #4, #5, #7, #8 — the entire `+fsx` ablation column |
| 3 | Generator NLL adds a one-sided "boost positives" signal on top of a label-blind preference loss, which can hurt on weak-validator tasks | **MEDIUM** | #3, #4, #7 |

The first two are structural (the loss is not computing what we think
it's computing). The third is a more subtle interaction effect that
needs an ablation to confirm.

---

## 1. Validator NLL is computed at the wrong position in the wrong sequence

In `g` mode the forward pass is on the **generator** sequence:

```python
# scripts/ranking_loss_ref.py: PairwiseDataset.__getitem__ (g-mode branch)
input_i = prompt_i + completion_i
enc_i = self.tokenizer(input_i, padding='max_length', truncation=True,
                       max_length=self.max_length, return_tensors='pt')
```

For persona-v1: `prompt_i = "Tell me something you would say:"`,
`completion_i = " " + statement`. So `input_ids_i` ends with the
**statement text**. There is no Yes/No answer slot anywhere in the
sequence. There is no discriminator prompt anywhere in the sequence.

But the validator-NLL term inside the `g`-mode training loop reads
Yes/No log-odds keyed off `token_correct_i` (= `" Yes"` / `" No"`):

```python
# scripts/ranking_loss_ref.py inside the train loop, g-mode branch
if validator_log_odds:
    logodds_correct_i = compute_logodds_simple(log_probs_i, token_correct_i)
    logodds_correct_j = compute_logodds_simple(log_probs_j, token_correct_j)
    nll_validator_loss = (
        pair_is_labeled * F.binary_cross_entropy_with_logits(logodds_correct_i, indicator_i) +
        pair_is_labeled * F.binary_cross_entropy_with_logits(logodds_correct_j, indicator_j)
    ).mean() / 2
else:
    score_correct_i = sum_completion_logprobs(log_probs_i, token_correct_i)
    score_correct_j = sum_completion_logprobs(log_probs_j, token_correct_j)
    nll_validator_loss = -(pair_is_labeled * (score_correct_i + score_correct_j)).mean() / 2
```

`compute_logodds_simple` indexes the logit tensor at
`pred_pos = -(comp_len + 1)`, where `comp_len` is the length of the
Yes/No tokens (typically 1). That's position −2 from the end of
`[gen_prompt][statement]` — i.e. **somewhere inside the statement,
where the model is predicting the last word of the statement**. The
model's P(Yes) vs P(No) at that position is meaningless for a
validator (Yes/No) task: it's whatever P(Yes) and P(No) happen to be
when the model is in the middle of producing a personality statement.

The `else` branch is the same flavor of broken:
`sum_completion_logprobs(log_probs_i, token_correct_i)` indexes the
*last* `comp_len` tokens of the sequence, so it's effectively asking
"how likely is the last token of the statement to literally be the
token ` Yes`?" instead of computing P(Yes) at the actual answer slot.

**Consequence.** The validator-NLL term is training on a
position/prompt mismatch every step, on every `comb` run. Whatever
gradient signal it produces is not what we wanted (BCE on the
discriminator's Yes-vs-No log-odds at the answer slot).

**Why other modes don't have this.** `train_g_or_d == 'both'` runs a
**separate** forward pass on the discriminator prompt + Yes/No tokens
(see the disc/gen forward passes in the `both` branch of the train
loop) and reads log-odds at the correct slot. But `both` mode is
explicitly blocked from using NLL weights:

```python
if train_g_or_d == 'both' and (nll_validator_weight > 0 or nll_generator_weight > 0):
    raise ValueError("NLL weights ... are not yet supported with --train_g_or_d both. ...")
```

So in practice no comb run today uses a correctly-positioned validator
NLL.

**How to verify.** Open a finished #4 persona-v1 wandb run (or
`grep "nll_validator_loss" ~/logs/<jobid>.out`) and inspect the scale
and trajectory of `train/nll_validator_loss`. Either:

- the loss is moving on a meaningful scale (the bug is firing and
  training is being actively perturbed in a wrong direction), or
- the loss is tiny / flat (the bug is firing but the gradient is
  effectively dead at that position; comb is then ~just preference +
  generator-NLL, which is its own concern — see #3).

Either way, the path is not training the validator-NLL objective the
research plan describes.

**Fix sketch.** Add a second forward pass on the discriminator prompt
inside the g-mode training loop, mirroring how `both` mode does it,
and compute `compute_logodds_simple` on **that** pass. This is a
non-trivial patch; cheaper alternative is to gate `nll_validator_weight`
to 0 unless `train_g_or_d in ('d', 'both')` (i.e. accept that comb in
g-mode = preference + generator-NLL, not + validator-NLL).

---

## 2. `--force-same-x` is a no-op for persona-v1

In `g` mode, fsx groups pair candidates by `p_train_tune.prompt`:

```python
# scripts/ranking_loss_ref.py, pair-construction, g-mode branch
if args.force_same_x:
    prompt_to_indices = defaultdict(list)
    for idx, z in enumerate(Z):
        prompt = z[0].prompt  # z[0] is p_train_tune
        prompt_to_indices[prompt].append(idx)
```

`p_train_tune` in g-mode is the **generator** prompt. For persona-v1
this is a single constant string for every item:

```python
# src/tasks/persona.py
GEN_PROMPT = "Tell me something you would say:"
NEG_GEN_PROMPT = "Tell me something you would never say:"

def make_prompt(item, style="generator", shots="zero", gen_response=None, neg=False, **kwargs):
    ...
    if style == "generator":
        prompt = NEG_GEN_PROMPT if neg else GEN_PROMPT
        completion = " " + statement
    ...
```

So all items collapse into a single bucket. Then
`itertools.combinations(indices, 2)` (with-fsx path) and
`itertools.product(indices, repeat=2)` filtered by `i[0] < i[1]`
(no-fsx path) produce the **same set of unordered pairs**. The
delta-filter is identical between the two branches. The only remaining
difference is the sampling step — `random.sample(pair_inds, total_samples)`
in the no-fsx path vs the proportional-by-group sampler in the fsx path
which, for one group, also reduces to "take `total_samples` of them" up
to RNG.

**Consequence.** For persona-v1, runs flagged `+fsx` (#3 / #4 / #5 /
#7 / #8) and runs flagged `-fsx` (#6 / #9 — and #1, #2 in the no-fsx
launcher line) are training on the *same pair pool*. Any apparent
delta between fsx and no-fsx variants for persona-v1 is RNG noise,
not a real fsx ablation.

**How to verify.** The pair-construction code prints these debug lines
when fsx is on:

```
============================================================
FORCE-SAME-X MODE (train_g_or_d=g)
============================================================
Found N unique prompts
```

For persona-v1 we expect `N == 1`. Skim a #3 / #4 train log
(`/datastor2/jdr/logs/40715.out` for persona-v1 9b-it #3) and confirm.

**Implication for the research-plan template.** The
[IMPORTANT-RESEARCH-PLAN.md §2 diagnostic shape](IMPORTANT-RESEARCH-PLAN.md)
lists fsx-related comparisons:

> - Does fsx contribute given new-loss + TC? → #4 vs #11 (self), #7 vs #12 (neg).
> - Does fsx contribute given TC alone? → #6 vs #5, #9 vs #8.
> - Does fsx alone help? → #10 vs #2.

For persona-v1 specifically, **none of these comparisons are
meaningful** because fsx has no effect at all. The fsx ablation needs
a task with multiple distinct generator prompts (e.g. ifeval-concat,
hypernym-bananas-to-dogs, ambigqa) before any comparison can land. We
should remove or asterisk these comparisons in the persona-v1 results
discussion.

**Note.** This is task-specific, not a code bug. The fsx machinery is
correct; it's just trivially satisfied for any single-x task.
Persona-v0 has the same property (same `GEN_PROMPT` constant).

---

## 3. Generator NLL adds a one-sided "boost positives" signal on top of a label-blind preference loss

### The setup

In `g` mode the forward pass produces a single `log_probs_i` over
`[gen_prompt][statement]`. Two losses on labeled pairs reuse it:

```python
# Preference (always on, both labeled and unlabeled):
score_i = sum_completion_logprobs(log_probs_i, token_id_i)           # log P(stmt_i | gen_prompt)
score_j = sum_completion_logprobs(log_probs_j, token_id_j)           # log P(stmt_j | gen_prompt)
preference_loss = -log σ(score_j - score_i)                           # winner = j, by validator order

# Generator NLL (labeled pairs only):
score_gen_i = sum_completion_logprobs(log_probs_i, token_gen_i)
score_gen_j = sum_completion_logprobs(log_probs_j, token_gen_j)
nll_generator_loss = -(pair_is_labeled
                       * (score_gen_i * indicator_i + score_gen_j * indicator_j)).mean() / 2
```

Two facts about `g` mode + persona:

1. `score_gen ≡ score` because `token_gen_i = " " + statement = token_id_i`.
   So both losses act on the **same scalar** (`log P(stmt | gen_prompt)`)
   for each item.
2. Pair ordering is by **validator** log P(Yes | x), NOT by gold label.
   `i` = lower validator score, `j` = higher.

### What each loss does to each item

For a labeled pair `(i, j)`:

- **Preference**: pushes `score(stmt_j)` UP and `score(stmt_i)` DOWN.
  Always. Regardless of labels.
- **Generator NLL**: pushes `score(stmt_i)` UP iff `indicator_i == 1`
  (gold positive); pushes `score(stmt_j)` UP iff `indicator_j == 1`.
  Never has a "push down on a negative" term (`indicator == 0` zeroes
  out the gradient).

### The four cases on a labeled pair

Let `+` = gold positive, `−` = gold negative.

| Case | i (lo-val) | j (hi-val) | Pref on `score(stmt_i)` | NLL on `score(stmt_i)` | Pref on `score(stmt_j)` | NLL on `score(stmt_j)` | Verdict |
|---|---|---|---|---|---|---|---|
| A | − | + | ↓ | 0 | ↑ | ↑ | both correct |
| B | + | + | ↓ | ↑ | ↑ | ↑ | fight on i |
| C | − | − | ↓ | 0 | ↑ | 0 | **pref pushes a negative up** |
| D | + | − | ↓ | ↑ | ↑ | 0 | **fight on i AND pref pushes a negative up** |

Cases A and B are fine in expectation — A is what we want, B is
mostly redundant (both push `score(stmt_j)` up).

Cases C and D are the awkward ones: the **preference loss is moving
score(stmt_j) — a gold-negative — up under the generator prompt**, and
nothing in the comb loss opposes that.

### Where this is most likely to bite

Cases C and D require the validator to misrank the pair vs the gold
label. The frequency depends on the validator's quality at init.
Persona-v1 baselines (Raw GenROC × 100, mean across 6 personas, from
[`metrics-from-scores/persona_v1_*all_gen_roc_table_cells.csv`](../metrics-from-scores/)):

| Base | Raw | self-TC | neg-TC |
|---|---|---|---|
| `gemma-2-9b-it` | 42.96 | 40.88 | 57.19 |
| `gemma-2-2b-it` | 48.56 | 57.40 | 75.82 |
| `gemma-2-2b` | 40.01 | 37.83 | 73.89 |

Raw is roughly chance for all three (40–49). That's the regime where
the validator is misranking a non-trivial fraction of pairs, so cases
C/D are common. This is exactly the regime where the comb-loss
asymmetry hypothesis would predict trouble.

### Why pref-only siblings might do better anyway

A reasonable hypothesis: with `pref-only` (#5/#6/#8/#9), there's no
NLL competing with preference, no one-sided "boost positives" signal,
and TC reweights pairs by typicality, which is a sharper corrective
than the half-asymmetric NLL. Comb's NLL ends up adding a
labeled-positive boost without the matching labeled-negative
suppression, which can shift the score distribution in a way that
hurts ranking metrics like GenROC even when individual positive
log-probs go up.

Note this is a hypothesis, not a proven cause. It needs an ablation
(below).

### What to actually check

1. **Disagreement rate.** From a finished #4 run's pair list (or rebuild
   it offline from the dataset + base validator), count
   `(indicator_i, indicator_j)` over delta-filtered labeled pairs. If
   cases C+D together exceed ~20–30% of labeled pairs, the
   comb-loss-asymmetry story is plausible. If C+D are rare, NLL is
   mostly redundant-but-fine and the persona pattern needs a different
   explanation.

2. **Direct ablation: comb − NLL.** Run the persona-v1 #4 recipe but
   with `nll_validator_weight=0, nll_generator_weight=0` (i.e.,
   pref-only with the same TC + fsx + vlo flags as #4). If that lifts
   performance to #5's level or above, NLL was the issue. One-job
   ablation per base.

3. **Direct ablation: comb − preference on disagreement pairs.** A
   smaller but more targeted check: filter pair construction to
   *only* labeled pairs where validator agrees with gold (drop case
   C/D pairs from the labeled set; keep all unlabeled pairs as
   normal). If #4 with this filter beats #4 without, preference on
   disagreement pairs is doing measurable damage.

### What this is NOT

- This is **not** "SFT toward labeled positives is wrong." That part is
  fine and is the intended behavior of the generator-NLL term in
  isolation.
- This is **not** a code bug — every line above does what it says. It's
  a structural interaction between the two loss terms when validator
  and labels disagree, plus the fact that NLL only acts on positives.
- This is **lower priority than #1 and #2** for explaining the persona
  ranking, because #1 means validator-NLL is firing on the wrong
  thing entirely, and #2 means half the structure of the experiment
  doesn't exist for persona. Both are testable in minutes; #3 needs an
  ablation run.

---

## What to do with this

In rough order:

1. **Check #2 first** — it's a one-grep verification. Confirm fsx is a
   no-op for persona-v1 in the train logs. If so, drop fsx from the
   persona-v1 results discussion entirely.

2. **Check #1 next** — read the wandb traces for one #4 run to see
   what `train/nll_validator_loss` is doing. If it's nonzero and
   moving, that's a real misdirected gradient on every comb step.

3. **Then consider the #3 ablation** — run a `pref-only + fsx + TC`
   variant of #4 (i.e. zero out NLL weights, keep everything else) on
   one base. Compare to #4 and #5. If it lands closer to #5 than #4,
   NLL is hurting; if it lands at #4, NLL is innocent and we need a
   different explanation for the persona ranking.

4. (Independent of the above) decide whether to fix the validator-NLL
   path for `g` mode. If the answer to #1 is "loss is small / flat",
   the cheapest fix is to gate `nll_validator_weight` to 0 in `g`
   mode and reframe the research plan: "comb in g-mode is preference
   + generator-NLL, not + validator-NLL." If we genuinely want
   validator-NLL during g-mode training, we need a second forward
   pass on the discriminator prompt — non-trivial patch.

---

# Locked design for `scripts/ranking_loss_ref_fix.py` (2026-05-21)

A separate `ranking_loss_ref_fix.py` was forked from `ranking_loss_ref.py`
to land the corrections from this analysis without disturbing existing
training. Scope: **only `g` mode is supported** in the fix. (`d` and
`both` modes are out of scope; if someone passes them, fall back to the
unmodified path or raise.)

## Locked decisions (answered 2026-05-21)

1. **NLL is per-item, not per-pair.** The `pair_is_labeled` gate is
   removed. Generator-NLL fires on every labeled **positive** item that
   appears in any pair; validator-NLL fires on every labeled item
   (modulo decision 6).
2. **Drop inconsistent pairs.** Preference loss only fires on pairs
   where every labeled item is on its "natural side" (see framework
   below). Inconsistent pairs are not down-weighted; they are excluded
   from pair construction.
3. **Drop case-B (same-class labeled) pairs.** L+/L+ and L−/L− pairs
   are dropped — preference would arbitrarily pick a winner among
   gold-equals based on validator noise.
4. **Single sampling stream** for the first version. (Two-stream
   refactor — preference pairs vs per-item NLL sweep — deferred.)
5. **Stratification by pair shape is a tunable knob.** Pair pool is
   partitioned by shape; sampler takes per-shape weights. Default:
   slightly oversample labeled-touching pairs (concrete value below).
6. **Validator-NLL position bug is out of scope.** In `g` mode the
   current val-NLL reads Yes/No log-odds at a position inside the
   statement (concern #1 of this doc). Fixing it requires a second
   forward pass on the discriminator prompt. For now, **`g`-mode
   val-NLL is hard-disabled** in the fix file (weight forced to 0 with
   a warning) until a separate refactor lands.
7. **Generator NLL stays one-sided.** Only fires for labeled positives;
   no negative-suppression term. (Adding `-log(1-P(stmt))` for
   negatives is unbounded and out of scope.)

## Simplified case framework

The cleanest mental model: each item has a **natural side** in any
pair `(i, j)` with `val(i) < val(j)`:

- **L+** (labeled positive) → belongs on **HI** (j)
- **L−** (labeled negative) → belongs on **LO** (i)
- **U** (unlabeled) → either side OK

A pair is **valid** iff every labeled item is on its natural side.
This collapses into 4 valid pair shapes (cross-product of allowed
LO and HI pools):

| LO pool | HI pool | shape       |
|---------|---------|-------------|
| L−      | L+      | `case_A`    |
| L−      | U       | `mixed_neg` |
| U       | L+      | `mixed_pos` |
| U       | U       | `both_U`    |

Equivalently: **invalid shapes** (which the current
`ranking_loss_ref.py` happily generates and trains on) are exactly:

- `(L−, L−)` → preference pushes one negative up
- `(L+, L+)` → preference picks a winner among gold-equals
- `(L+, L−)` → preference suppresses positive AND boosts negative
- `(L+, U)` → preference suppresses a known positive
- `(U, L−)` → preference boosts a known negative

Under the natural-side filter, every loss term either agrees with
gold on labeled items or defers to validator on unlabeled items.
There is no item where preference and gen-NLL want opposite things.

## Loss formulas (per pair, post-filter)

Let `s(item) = log P(stmt | gen_prompt)` be the generator score for
the item's statement, computed from one forward pass of the current
model on `[gen_prompt][stmt]`. For each kept pair `(i, j)` with
`val(i) < val(j)`:

```
pref_loss   = -log σ( s(j) − s(i) )

gen_nll     = - ( 1[label(i) == pos] · s(i)
                + 1[label(j) == pos] · s(j) ) / 2

val_nll     = 0     # disabled in g-mode per decision 6

per_pair_loss = w_pref · pref_loss
              + w_gen  · gen_nll
              + w_val  · val_nll
```

Then `total_loss = mean over batch of per_pair_loss`.

Notes on normalization:

- `gen_nll` is divided by 2 to match the existing `nllg=1.0` scale
  (current code uses `(score_gen_i * indicator_i + score_gen_j *
  indicator_j) / 2`). With the per-item framing the `indicator` is
  exactly `1[label == pos]` per side; for unlabeled items it is 0.
- The previous `pair_is_labeled` outer gate (which zeroed the entire
  NLL contribution unless **both** items were labeled) is removed.
  Pairs where exactly one side is labeled now contribute partial NLL
  signal on that side, which is the whole point of the fix.

## Pair-construction algorithm

```python
def build_pair_pool(items, val_scores, labels, delta,
                    shape_weights, total_samples, rng):
    """
    Returns: list of (i, j) index pairs, val(i) < val(j), |Δval| > delta,
             with every labeled item on its natural side.

    items, val_scores, labels: parallel arrays of length N.
    labels[k] ∈ {'pos', 'neg', 'unlabeled'}.
    shape_weights: dict mapping shape name -> non-negative weight.
                   Normalized to sum to 1 internally.
    """
    L_pos = [k for k, lab in enumerate(labels) if lab == 'pos']
    L_neg = [k for k, lab in enumerate(labels) if lab == 'neg']
    U     = [k for k, lab in enumerate(labels) if lab == 'unlabeled']

    def enumerate_shape(lo_pool, hi_pool, allow_both_lo_hi=True):
        """Build all (i, j) with val(i) < val(j), |Δval| > delta,
        i ∈ lo_pool, j ∈ hi_pool. Skip i == j when pools overlap.
        """
        pairs = []
        for i in lo_pool:
            for j in hi_pool:
                if i == j:
                    continue
                if val_scores[i] < val_scores[j] and \
                   (val_scores[j] - val_scores[i]) > delta:
                    pairs.append((i, j))
        return pairs

    pool_by_shape = {
        'case_A':    enumerate_shape(L_neg, L_pos),
        'mixed_neg': enumerate_shape(L_neg, U),
        'mixed_pos': enumerate_shape(U,     L_pos),
        'both_U':    enumerate_shape(U,     U),
    }

    # Stratified sampling per shape weights.
    total_w = sum(shape_weights.values())
    sampled = []
    for shape, w in shape_weights.items():
        target = round(total_samples * w / total_w)
        avail  = pool_by_shape[shape]
        take   = min(target, len(avail))
        if take > 0:
            sampled.extend(rng.sample(avail, take))

    # If we didn't hit total_samples (shape was empty / undersized),
    # backfill from any non-exhausted shape, weighted same way.
    deficit = total_samples - len(sampled)
    if deficit > 0:
        flat = [(s, p) for s, lst in pool_by_shape.items() for p in lst]
        # remove already-sampled
        already = set(sampled)
        flat = [(s, p) for (s, p) in flat if p not in already]
        if flat:
            sampled.extend([p for (_s, p) in rng.sample(flat,
                                                        min(deficit, len(flat)))])

    rng.shuffle(sampled)
    return sampled, pool_by_shape  # second return for diagnostics/logging
```

Default `shape_weights` for the fix file (knob 5):

```python
DEFAULT_SHAPE_WEIGHTS = {
    'case_A':    0.20,   # ~20% of budget — high signal, small pool
    'mixed_neg': 0.20,
    'mixed_pos': 0.20,
    'both_U':    0.40,   # ~40% of budget — large pool, no labels
}
```

These are weights, not hard floors. If a shape's pool is smaller than
its target, the deficit is backfilled from non-exhausted shapes
(uniformly weighted by their full pool size). All four weights are
expose as CLI flags for ablation.

For comparison, **random** sampling under the current code produces
pair shapes in proportion to their pool sizes — for `semi 0.1` on
balanced data that's roughly 0.5% case_A / 9% mixed_neg / 9%
mixed_pos / 81% both_U, so the default above is a meaningful shift
toward labeled-touching pairs.

## Pair-count estimate for persona-v1 (sanity check, decision 5 feasibility)

Persona-v1 train.csv has **N = 1500** items, perfectly balanced
(750 yes / 750 no), 500 per persona × 3 personas. With
`--semi-supervised 0.1` (10% labeled) and balanced sampling:

| Set   | Size  |
|-------|-------|
| L+    | ~75   |
| L−    | ~75   |
| U     | ~1350 |

Pair pool sizes (before val-orientation and delta filters):

| Shape       | Unordered count |
|-------------|-----------------|
| `case_A`    | 75 × 75 = 5,625 |
| `mixed_neg` | 75 × 1350 = 101,250 |
| `mixed_pos` | 75 × 1350 = 101,250 |
| `both_U`    | 1350 × 1349 / 2 = 910,575 |
| **total**   | **1,118,700** |

**Measured pool sizes** (via `scripts/_smoke_fix1_pairs.py`, synthetic
val-scores: L+ ~ N(0.5, 1), L− ~ N(−0.5, 1), U ~ N(0, 1), delta=0.15,
75/75/1350 partition, seed=42):

| Shape       | After all filters | Default budget @ 5110 total |
|-------------|-------------------|----------------------------|
| `case_A`    | **4,027** | 0.20 × 5110 = 1,022 |
| `mixed_neg` | **56,447** | 1,022 |
| `mixed_pos` | **63,146** | 1,022 |
| `both_U`    | **835,903** | 2,044 |
| **total**   | **959,523** | 5,110 |

**Verdict.** All four shapes have ample pool size for the default
stratified budget. `case_A` is the smallest pool but still ~4× the
default budget for that shape; the fix is comfortably feasible with
the default weights even when the validator is only modestly
informative.

> ⚠ Caveat: synthetic val-scores assume a moderately informative
> validator (mean separation = 1.0 between L+ and L− distributions in
> log-prob space). Real `gemma-2-9b-it` baselines on persona-v1 are
> closer to chance (Raw GenROC = 42.96), so the case_A pool may be
> smaller in practice. Even halving it to ~2,000 leaves the default
> budget reachable. If a base has so weak a validator that the case_A
> pool drops below the budget, the backfill logic in
> `ranking_loss_ref_fix.py` redistributes from the other shapes.

## Implementation plan for `ranking_loss_ref_fix.py`

Targeted edits (no rewrite):

1. **Add a g-mode-only guard** near the top of `main()` after
   `train_g_or_d` is parsed. Raise on `d` / `both`.
2. **Replace pair construction** (g-mode branch, lines ~1696–1788 in
   the original `ranking_loss_ref.py`) with `build_pair_pool` above.
   Keep the `Z` / `pairs_` data structure that the downstream
   serializer (lines ~1916–1944) consumes; only the **pair index
   selection** changes. Print pool diagnostics (sizes per shape,
   sampled per shape).
3. **Remove `pair_is_labeled` gate** from gen-NLL inside the train
   loop. Replace with explicit `1[label == pos]` per item, derived
   from `indicator_i` / `indicator_j` (which already encode this in
   the dataset class).
4. **Hard-disable val-NLL in g-mode**. In the train loop, force
   `nll_validator_loss = 0.0` and emit a one-time warning if the user
   passes a non-zero `--nll_validator_weight` in g-mode.
5. **Add CLI flags** for shape weights:
   `--shape-weight-case-a`, `--shape-weight-mixed-neg`,
   `--shape-weight-mixed-pos`, `--shape-weight-both-u`. Default to the
   table above. Validate they sum to >0.
6. **Update model-name suffix** to encode the change so eval files
   don't collide with the original. Add a `_fix1` token to the model
   short name (similar to existing `_merged`, `_force-same-x` tokens)
   so trained checkpoints and score CSVs are unambiguously from the
   fixed code path.

Touch the **dataset class** (`PairwiseDataset.__getitem__`) **only if
needed**; the current 7-tuple structure already carries `indicator`
and `is_labeled`, which is everything the fixed loss needs. Keep
`pair_is_labeled` in the batch dict for backward compat / diagnostics
even if unused.

## What this fix does NOT cover

- Validator-NLL position correctness (concern #1). Deferred per
  decision 6.
- `--force-same-x` is a no-op for persona-v1 (concern #2). This is a
  task-property issue, not a code bug; not in scope of any code fix.
  Persona results should drop fsx-vs-no-fsx comparisons.
- Two-stream sampling (independent NLL pass over all labeled items).
  Deferred per decision 4.
- `d` mode and `both` mode. Out of scope; the fix file raises on
  them.
