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
