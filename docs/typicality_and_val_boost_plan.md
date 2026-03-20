# Typicality Correction & Validator Boost Implementation Plan

## Current Code Architecture

### Modes
- **`'d'` (g2d)**: Train discriminator from generator signal. Generator scores used for pair selection.
- **`'g'` (d2g)**: Train generator from validator signal. Validator scores (log P("Yes")) used for pair selection.
- **`'both'`**: Train both. Uses tuples `(gen_score, val_score)` in `logprobs_last_layer`. Pair selection sorts by generator score (index 0).

### Score Types
- **Generator score**: log P(completion | prompt) — probability of the generator completion
- **Validator score**: log P("Yes" | prompt) — probability of "Yes" token (or log-odds if `--validator-log-odds`)

### Training Flow

1. **Pre-compute scores** → stored in `logprobs_last_layer`
   - `'d'`: generator scores only
   - `'g'`: validator scores only
   - `'both'`: tuples of (generator, validator)

2. **Pair selection** (OFFLINE) → based on `logprobs_last_layer`
   - Pairs selected based on score differences > delta

3. **Training loop** (ONLINE) → fresh forward pass each batch
   - `score_i`, `score_j` computed from current model state
   - Loss computed from these fresh scores
   - Gradients flow through model

### Current Typicality Correction (lines ~1121-1166)
- Only applies to `'d'` and `'both'` modes
- Pre-computes GPT-2's unconditional P(completion) for each example
- Subtracts from `logprobs_last_layer` (affects pair selection only)
- Does **NOT** affect online scores in training loop

---

## Problem 1: Typicality Correction is Incomplete

### Issue
Typicality correction currently:
1. Only applies to `'d'` and `'both'` modes — NOT `'g'` mode
2. Only affects **pair selection** scores, NOT **training loop** scores

### Why we can't just enable it for `'g'` mode
In `'g'` mode, `logprobs_last_layer` contains **validator scores** (log P("Yes")), not generator scores.

The current code does:
```python
logprobs_last_layer[i] -= typicality_scores[i]
```

If enabled for `'g'` mode, this would subtract GPT-2's P(completion) from the validator's P("Yes") — **nonsensical!**

### Desired Behavior
1. Typicality correction should ONLY affect **generator scores**
2. In `'g'` mode: generator scores are computed fresh in training loop (not used for pair selection)
3. So we need to apply typicality correction **during training**, not during pre-computation
4. This should apply to ALL modes

---

## Problem 2: Add `--boost-initial-val` Flag

### Goal
Shift validator scores so optimal classification threshold = 0

### Desired Behavior
1. Before training: compute validator scores for all training examples
2. Find optimal threshold `t*` that maximizes accuracy (pos/neg classification)
3. Compute `theta = -t*`
4. During training loop: `boosted_val_score = fresh_val_score + theta`

---

## Implementation Plan

### Data Structure Change

Currently:
```python
Z = list(zip(L_train_all, p_train_gold_list, logprobs_last_layer))
# Each Z[i] = (L_item, prompt_item, precomputed_score)
```

Proposed:
```python
Z = list(zip(L_train_all, p_train_gold_list, logprobs_last_layer, typicality_scores))
# Each Z[i] = (L_item, prompt_item, precomputed_score, typicality_score)
```

This way, typicality is bundled with each example and available during batching.

---

## TODO Checklist

### Typicality Correction

- [x] **1. Pre-compute typicality scores for ALL modes**
  - Compute `typicality_scores` regardless of mode
  - Only subtract from `logprobs_last_layer` in `'d'` and `'both'` modes (for pair selection)
  - Keep `typicality_scores` list intact for use in training loop

- [x] **2. Include typicality scores in Z tuple**
  - Add as 4th element: `(L_item, prompt_item, precomputed_score, typicality_score)`
  - If typicality disabled, use `[0.0] * len(...)` as placeholder

- [x] **3. Update PairwiseDataset / training loop**
  - Unpack typicality from each item in pair
  - Apply correction to generator scores: `corrected_gen_score = fresh_gen_score - typicality`
  - This applies in ALL modes (`'g'`, `'d'`, `'both'`)

### Validator Boost

- [x] **4. Add `--boost-initial-val` argument**
  - `parser.add_argument("--boost-initial-val", action='store_true', ...)`

- [x] **5. Compute optimal threshold for validator boost**
  - Pre-compute validator scores for all training examples
  - Get ground truth labels (pos=1, neg=0) from `L_train_all`
  - Use sklearn `roc_curve` to find threshold maximizing accuracy
  - Store `val_boost_theta = -optimal_threshold`

- [x] **6. Apply val_boost_theta in training loop**
  - Add `val_boost_theta` to fresh validator scores
  - This is a single constant, same for all examples

---

## Implementation Notes

- Typicality correction targets **generator scores only** (never validator)
- Val boost targets **validator scores only**
- Both corrections apply during the **training loop** (online), not just pair selection
- Val boost theta is a single scalar; typicality varies per example

## Changes Made (Completed 2026-01-19)

1. **Imports**: Added `from sklearn.metrics import roc_curve`

2. **Helper function**: Added `compute_optimal_threshold(scores, labels)` to find threshold maximizing accuracy

3. **Typicality computation**: Now computes for ALL modes, but only modifies `logprobs_last_layer` for 'd'/'both' modes

4. **Z tuple**: Extended to 4 elements including typicality scores

5. **Pairs structure**: Extended to include typicality scores (6th element for 'd'/'g', 4th for 'both')

6. **PairwiseDataset**: Updated `__getitem__` to return `typicality_i` and `typicality_j` tensors

7. **Training loop typicality**: Applied correction to generator scores in ALL modes

8. **Val boost computation**: Computes optimal threshold before training (handles all modes)

9. **Training loop val boost**: Applied to validator scores in 'd' and 'both' modes
