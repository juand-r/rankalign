# Semi-Supervised Training Plan

## Overview

We split training prompts (questions) into **labeled** and **unlabeled** subsets at the prompt level.
- **Labeled pairs**: use whatever loss weights are specified (comb, sft, or pref)
- **Unlabeled pairs**: always get preference loss at weight 1.0, NLL terms zeroed out

The split is deterministic given `--split-seed` (default 42). Both `--semi-supervised` and `--labeled-only` use the same seed, so the labeled set is identical across comparable runs.

## Experimental Settings

| Setting | Flags | Model name contains |
|---|---|---|
| SS comb + pref (10% labeled) | `--semi-supervised 0.1` + comb flags | `-semi0.1` |
| SS sft + pref (10% labeled) | `--semi-supervised 0.1` + sft flags | `-semi0.1` |
| Labeled-only comb (10%) | `--labeled-only 0.1` + comb flags | `-labelonly0.1` |
| Labeled-only sft (10%) | `--labeled-only 0.1` + sft flags | `-labelonly0.1` |
| Labeled-only pref (10%) | `--labeled-only 0.1` + pref flags | `-labelonly0.1` |
| Full pref (all data) | pref flags, no semi flags | (existing, no new marker) |
| Full comb (all data) | comb flags, no semi flags | (existing, no new marker) |

### Key comparisons

- **Does unlabeled data help?** Compare SS comb vs Labeled-only comb (same labeled set, but SS also trains on unlabeled pairs with pref loss)
- **Does unlabeled data help (SFT)?** Compare SS sft vs Labeled-only sft
- **How much does labeling matter?** Compare Labeled-only comb/sft/pref (10%) vs Full comb/pref (100%)

### Loss type reminders

- **comb** = preference + NLL validator + NLL generator (`--nll_validator_weight 1.0 --nll_generator_weight 1.0`)
- **sft** = NLL only, no preference (`--preference_loss_weight 0.0 --nll_validator_weight 1.0 --nll_generator_weight 1.0`)
- **pref** = preference only (`--nll_validator_weight 0.0 --nll_generator_weight 0.0`)

## Implementation details

- Split happens at the **prompt level** (all answers for a given question are either labeled or unlabeled)
- For `--labeled-only`: unlabeled items are discarded before pair formation (smaller training set)
- For `--semi-supervised`: all items kept, `is_labeled` flag propagated to pairs; NLL terms multiplied by `pair_is_labeled` (1.0 if both items in pair are labeled, 0.0 otherwise); unlabeled pairs always get preference loss at weight 1.0 regardless of `--preference_loss_weight`
- `--semi-supervised` and `--labeled-only` are mutually exclusive (raises error if both given)
- Logprobs are computed for all items before the split (slight waste for `--labeled-only`, but one-time startup cost)
- `total_samples` (number of pairs) is applied after the split
