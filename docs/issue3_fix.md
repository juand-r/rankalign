# Issue #3 fix: per-prompt × per-shape pair sampling

This is the design doc for the actual fix in
`scripts/ranking_loss_ref_fix.py` for **Issue #3** of
[`docs/comb_loss_g_mode_concerns.md`](comb_loss_g_mode_concerns.md):
generator-NLL conflicts with preference loss on inconsistent pairs.

The earlier design discussion in `comb_loss_g_mode_concerns.md` covers
the *why*. This doc is the *what* and *how* — the algorithm we
actually implement.

## Scope

- Only `--train_g_or_d g` is supported. d / both / iter raise.
- Validator-NLL **position** bug (concern #1) is out of scope: the
  Yes/No log-odds are still read at the same (incorrect-for-g-mode)
  position the parent uses. Fixing the position needs a 2nd forward
  pass on the discriminator prompt and will be a separate fix.
- Validator-NLL **gating**, however, is changed: matched to gen NLL,
  it now fires per item (every labeled item), not per pair. See
  "Loss block" below.
- `--force-same-x` no-op-on-persona observation (concern #2) is
  out of scope; fsx behaves identically to the parent except now
  composes with the 4-shape filter.

## Definitions

Each item in `Z` (the sorted training data) has:
- a **prompt** (`z[0].prompt`) — the generator prompt string,
- a **validator score** `val(k)` (= `z[1]`),
- a **gold label class** in `{L_pos, L_neg, U}` from
  `_label_class_for_z`:
  - `L_pos` = labeled, gold positive,
  - `L_neg` = labeled, gold negative,
  - `U`     = unlabeled.

A **pair** `(i, j)` is "consistent" iff:
1. `val(i) < val(j)` and `val(j) − val(i) > delta`, AND
2. every labeled item is on its **natural side**: `L_pos` on the
   HI side `j`, `L_neg` on the LO side `i`.

There are exactly **4 valid pair shapes**:

| Shape      | LO pool (lower val) | HI pool (higher val) |
|------------|---------------------|----------------------|
| `case_A`   | `L_neg`             | `L_pos`              |
| `mixed_neg`| `L_neg`             | `U`                  |
| `mixed_pos`| `U`                 | `L_pos`              |
| `both_U`   | `U`                 | `U`                  |

Pairs from any other shape (same-class labeled, validator-disagrees-
with-gold, etc.) are dropped at construction time.

## Algorithm

```
INPUTS:
  Z                  : sorted training data (by val score, ascending)
  delta              : margin filter on validator scores
  total_samples      : global pair budget
  shape_weights      : {case_A: 0.20, mixed_neg: 0.20,
                        mixed_pos: 0.20, both_U: 0.40}   (CLI flags)
  args.force_same_x  : whether to constrain pairs to same prompt

# 1. Group items by prompt.
# When fsx is OFF, treat the entire dataset as one virtual group.
if args.force_same_x:
    prompt_groups = {prompt_str: [indices in Z with this prompt]}
else:
    prompt_groups = {None: list(range(len(Z)))}

# 2. Per-prompt × per-shape enumeration.
# Each (prompt, shape) cell gets its own list of (i, j) index pairs,
# already filtered by delta inline.
N_total = len(Z)
pool[prompt][shape] = []   # 4 entries per prompt
prompt_n[prompt]    = 0    # number of completions for this prompt
for prompt, indices in prompt_groups.items():
    grp_lpos, grp_lneg, grp_u = partition(indices)        # by label class
    pool[prompt]['case_A']    = enumerate_shape(grp_lneg, grp_lpos)
    pool[prompt]['mixed_neg'] = enumerate_shape(grp_lneg, grp_u)
    pool[prompt]['mixed_pos'] = enumerate_shape(grp_u,    grp_lpos)
    pool[prompt]['both_U']    = enumerate_shape(grp_u,    grp_u)
    prompt_n[prompt]          = len(indices)

# 3. Per-prompt budget allocation, proportional to completion count.
# Each completion gets equal expected exposure to training, regardless
# of which prompt it lives in (see "Why proportional-to-points" below).
for prompt:
    prompt_budget[prompt] = round(total_samples * prompt_n[prompt] / N_total)

# 4. Sample per (prompt, shape).
sampled = []
for prompt:
    prompt_sampled = []
    for shape, w in shape_weights.items():
        target = round(prompt_budget[prompt] * w / sum(shape_weights.values()))
        avail  = pool[prompt][shape]
        take   = min(target, len(avail))
        if take > 0:
            prompt_sampled.extend(random.sample(avail, take))

    # 5. Within-prompt backfill: if rounding / empty shapes left a
    # deficit relative to prompt_budget, fill from this prompt's
    # remaining pool (any shape, weighted uniformly by leftover size).
    deficit = prompt_budget[prompt] - len(prompt_sampled)
    if deficit > 0:
        leftover = [(shape, p) for shape in pool[prompt]
                               for p in pool[prompt][shape]
                               if p not in prompt_sampled]
        prompt_sampled.extend(random.sample(leftover, min(deficit, len(leftover))))

    sampled.extend(prompt_sampled)

random.shuffle(sampled)
return sampled
```

## Why proportional-to-points

Allocating `prompt_budget[p] = total_samples * n_p / N_total` (where
`n_p` is the number of completions for prompt `p` and `N_total` is the
total completion count) gives every completion **equal expected
exposure** to training:

```
E[exposures(c)] = 2 * prompt_budget[p] / n_p
                = 2 * total_samples * (n_p / N_total) / n_p
                = 2 * total_samples / N_total
                = constant across all completions
```

Alternative rules and why we don't use them:

- **Equal split per prompt** (parent's fair-share): a completion in a
  10-item prompt is exposed 10× more than a completion in a 100-item
  prompt. Biased toward small prompts.
- **Proportional to pair count** (`n_p² / sum(n_q²)`): big prompts
  dominate quadratically. A completion in a 100-item prompt is exposed
  10× more than one in a 10-item prompt. Biased toward big prompts.
- **Proportional to points** (this rule): each completion contributes
  equally to gradients in expectation. Neutral.

For tasks with a single generator prompt (e.g. persona-v1), all three
rules collapse to the same allocation.

## Backfill policy: within-prompt only

When a `(prompt, shape)` cell has fewer pairs than its target, the
deficit is redistributed within that prompt to its other shapes
(uniformly weighted by the leftover pool size, NOT by the original
shape weights — the leftover is whatever survived after first-pass
sampling).

Two alternative backfill strategies were considered but **not
implemented**:

- **No backfill** — leaves `total_sampled < total_samples` whenever
  any cell has a deficit. Cleanest but loses budget on multi-prompt
  tasks where many small prompts have empty shapes.
- **Within-shape (cross-prompt)** — preserves the global per-shape
  ratio at the cost of breaking the per-prompt budget allocation
  (one prompt's leftover `case_A` budget would go to a different
  prompt). Breaks the "every completion equally exposed" property.

Within-prompt backfill preserves the per-prompt budget exactly and
keeps the global per-shape ratio approximate (within rounding +
backfill noise). For the project's current tasks this is the right
trade.

> TODO: if a future task hits a case where many prompts have empty
> shapes (e.g. ~50% of prompts contain only L_pos items), and the
> global per-shape ratio drifts visibly from the requested weights,
> revisit and add `--shape-backfill {within-prompt, within-shape, none}`
> as a CLI flag.

## Loss block

```
preference_loss  = -log σ(score_j - score_i)               # every pair
gen_NLL_per_item = -score_gen * (is_labeled * indicator)   # per item; fires on L+
val_NLL_per_item = -P(correct | prompt) * is_labeled       # per item; fires on L+ AND L-
                   (or BCE-with-log-odds version when --validator-log-odds is on)
loss = w_pref * preference_loss + w_g * gen_NLL_per_item + w_v * val_NLL_per_item
```

Per-shape effective contribution:

| Shape       | preference | gen NLL fires | val NLL fires |
|-------------|------------|---------------|---------------|
| `case_A`    | yes        | on j (L+)     | on i (L−) AND j (L+) |
| `mixed_pos` | yes        | on j (L+)     | on j (L+) |
| `mixed_neg` | yes        | nowhere       | on i (L−) |
| `both_U`    | yes        | nowhere       | nowhere |

Both NLLs are gated by `is_labeled_*` per item. Gen NLL additionally
multiplies by `indicator_*` so it only fires for labeled **positives**
(this is the "one-sided" property — we want to boost positives, not
suppress negatives via gen NLL). Val NLL fires for both labeled
positives and labeled negatives — BCE handles the asymmetry correctly
(positive items pull `Yes` up, negative items pull `No` up).

## Files

- [`scripts/ranking_loss_ref_fix.py`](../scripts/ranking_loss_ref_fix.py)
  — implementation
- [`scripts/_smoke_fix1_pairs.py`](../scripts/_smoke_fix1_pairs.py)
  — smoke test (verifies per-prompt budget allocation, fsx
  invariants, no invalid pairs)
- [`docs/comb_loss_g_mode_concerns.md`](comb_loss_g_mode_concerns.md)
  — original analysis (the *why*); see "Locked design" section for
  the higher-level history of the decisions captured here.
