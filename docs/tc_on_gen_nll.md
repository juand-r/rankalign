# Should typicality correction apply to gen-NLL?

This doc resolves a recurring ambiguity in `comb`-style training (preference
+ gen-NLL + val-NLL): when typicality correction (TC) is on, do we apply it
only to the preference loss (current code), or also to the gen-NLL term?

The answer depends on whether TC is **offline** (precomputed, frozen) or
**live-with-grads** (recomputed each step against the trainable model).

The current `scripts/ranking_loss_ref_fix.py` is **offline only**. Live TC is
a future direction discussed in `docs/IMPORTANT-RESEARCH-PLAN.md` §3(a) and
§7.

---

## Setup

For a training item, define:

- `s(item) = log P_θ(stmt | gen_prompt)` — generator score (depends on θ).
- `t(item)` — typicality reference. Two flavors in the code:
  - **self-TC**: `log P(stmt | null_context)`.
  - **neg-TC**: `log P(stmt | neg_prompt)`.
- `s_corrected(item) = s(item) − t(item)`.

In the **offline** implementation (current code), `t(item)` is precomputed
once before training under `torch.no_grad()` and stored as a Python `float`.
It is θ-independent for the entire run.

In the **live-with-grads** proposal (not implemented), `t(item)` is computed
each step against the *current* trainable model with grads enabled, so
`∇_θ t(item) ≠ 0`.

---

## Offline TC: applying TC to gen-NLL is a no-op

In the current code:

```python
# Apply typicality correction to generator scores (preference loss).
if args.typicality_correction and train_g_or_d == 'g':
    score_i = score_i - typicality_i
    score_j = score_j - typicality_j

# Preference loss uses TC-corrected scores.
diff = score_j - score_i
preference_loss = -torch.log(torch.sigmoid(diff))

# Gen-NLL uses RAW score_gen_*, not TC-corrected.
score_gen_i = sum_completion_logprobs(log_probs_i, token_gen_i)
score_gen_j = sum_completion_logprobs(log_probs_j, token_gen_j)
nll_generator_loss = -(gen_w_i * score_gen_i + gen_w_j * score_gen_j).mean()
```

If we instead applied TC to gen-NLL:

```
gen_NLL_TC  = -(s(item) - t(item))  = -s(item) + t(item)
```

The gradient w.r.t. θ:

```
∇_θ gen_NLL_TC = -∇_θ s(item) + ∇_θ t(item)
              = -∇_θ s(item) + 0       (t is frozen offline)
              = ∇_θ gen_NLL_raw
```

Identical optimizer steps. Adding TC to gen-NLL just shifts the loss value
by a frozen constant per item; training is bit-identical.

### Why does TC matter for preference but not gen-NLL?

Preference loss is softmax-flavored:
`preference = -log σ(s(j) − s(i))`. With TC:

```
diff_TC  = (s(j) - t(j)) - (s(i) - t(i)) = diff_raw - (t(j) - t(i))

∇_θ preference_TC = (1 - σ(diff_TC)) · ∇_θ[s(j) - s(i)]
```

The constant `−(t(j) − t(i))` shifts the operating point of σ, which
changes the **scalar weight** `(1 − σ)` per pair. Gradient direction is
unchanged but per-pair magnitude is reweighted by typicality. (This is the
"TC is a per-pair reweighter" point in IRP §3(a).)

Gen-NLL is a linear scoring loss: `-s(item)`. Additive constants pass
through `∇_θ` as zero. So the TC offset has zero effect on the gradient.

### Conclusion (offline)

Keep gen-NLL on raw scores. The current code is correct as written.
"Applying TC to gen-NLL" in the offline regime would just rescale the
reported loss value while leaving the model identical at every step.

This explains why the question is silent in the existing literature on
preference + SFT joint training: under offline reference subtraction, the
SFT term is reference-invariant.

---

## Live-with-grads TC: TC on gen-NLL is no longer trivial

Once `t(item)` flows gradients (`∇_θ t(item) ≠ 0`), the analysis flips.

### Without TC on gen-NLL (current behavior carried over to live regime)

```
∇_θ gen_NLL_raw = -∇_θ s(item)
```

Pushes `s(item)` up. Pure SFT toward the labeled positive's statement
under the gen prompt. Does not directly touch `t(item)`.

### With TC on gen-NLL

```
∇_θ gen_NLL_TC = -∇_θ s(item) + ∇_θ t(item)
```

Pushes `s(item)` up AND pushes `t(item)` DOWN (because the loss is
`-s + t`, and we're minimizing). For self-TC, this means making the
labeled positive statement *less* likely under null context while making
it *more* likely under the gen prompt. The model is being explicitly
shaped to make this statement "specific to" the gen prompt rather than
something it would say spontaneously.

### Two coherent positions

#### Position A — "TC is the metric we optimize; apply it everywhere"

If we believe the TC-corrected score `s − t` is the right objective
(matches eval, isolates conditional knowledge from prior), then every
loss term that takes a per-item score should use `s_corrected`. Both
preference loss and gen-NLL would optimize the same metric. Live TC on
gen-NLL is the principled extension of live TC on preference.

Behaviorally, gen-NLL becomes the per-item analog of preference: instead
of "rank winner above loser on the corrected scale," it is "score the
labeled positive on the corrected scale." For a labeled positive `j`,
both losses push the same way: `s(j) ↑`, `t(j) ↓`.

Cost: the degeneracy pressure on the null-context distribution `P_θ(·∣null)`
doubles. Preference loss already drives `t(j) ↓` for winners and `t(i) ↑`
for losers; gen-NLL adds another `t(j) ↓` term for every labeled positive
that appears as a winner. Without a regularizer, the model can
cheaply minimize loss by making `P_θ(·∣null)` arbitrarily sparse/weird.
A KL-to-base or `‖log P_θ(y) − log P_base(y)‖²` regularizer (IRP §7) is
mandatory in this setting, with stronger β than would be needed if only
preference loss had live TC.

#### Position B — "TC is a method-specific reweighting on preference; gen-NLL stays pure SFT"

If we view TC as a tool that lives specifically inside the pairwise
ranking objective (per-pair reweighter, IRP §3a in the offline case;
per-pair gradient redirector in the live case), then gen-NLL is not a
ranking objective and does not need TC. Gen-NLL remains "memorize this
labeled positive when prompted" — vanilla SFT. The two terms have
different jobs:

- Preference: rank items by `s − t` (corrected metric).
- Gen-NLL: produce labeled positives when prompted (raw conditional).

Cost: less coherent objective. For a labeled positive `j` that is a
winner in some pair, preference is pushing `t(j) ↓` while gen-NLL is
pushing `s(j) ↑` without touching `t(j)` — but if the model satisfies
gen-NLL by making `s(j)` go up "broadly" (i.e., raising both `s(j)` and
`t(j)` together), preference's `t(j) ↓` term has to fight harder.
The two terms can work at slight cross-purposes on `t(j)`. In practice
this effect is bounded because gen-NLL only fires on labeled positives
(~5% of training pairs in semi 0.1), so the cross-purpose is weak.

Benefit: half the degeneracy pressure on the null distribution, weaker
regularizer suffices.

### Recommendation for live TC (when we get there)

1. Implement live TC behind a flag (`--online-typicality` per IRP §7
   plan) so we can ablate against the current offline behavior.
2. **Default Position B** (gen-NLL stays raw) for the first live runs.
   Reason: it isolates "live TC on the ranking objective" as a single
   change vs. offline; we can compare to offline-TC + raw-gen-NLL
   (current code) cleanly, and only one of the two terms is gradient-
   degenerating the null distribution, so a milder regularizer
   should suffice.
3. **Add a flag for Position A** (`--tc-on-gen-nll` or similar) so we
   can ablate it. If empirically Position A's stronger explicit
   "push null down on positives" signal helps once the regularizer is
   tuned, switch the default. Don't run Position A without a
   regularizer.
4. Whichever position we pick, eval at both `--self-typicality` /
   `--neg-typicality` AND `--base-typicality` so we can detect
   null-distribution degeneracy (per IRP §3(a) "(May 10) Findings").

### Decision matrix

| Setting | TC on preference | TC on gen-NLL | Notes |
|---|---|---|---|
| Offline (current) | reweight (matters) | no-op | Current `_fix.py`. Don't change. |
| Live, Position A | gradient redirector | gradient redirector | Coherent objective, needs strong regularizer. |
| Live, Position B | gradient redirector | none (raw SFT) | Cleaner ablation, milder regularizer. |

---

## Implementation pointers

For the offline regime no change is needed: gen-NLL on raw scores is
correct.

For the live regime (when implemented):

- **Position B** is a one-line change: continue computing `score_gen_*`
  as `sum_completion_logprobs(log_probs_*, token_gen_*)` — but ensure
  `t(item)` is computed live with grads for the preference path
  (separate from gen-NLL).
- **Position A** would compute `s_gen_corrected = score_gen_* −
  typicality_*_live` and feed that into the gen-NLL term:
  ```python
  nll_generator_loss = -(gen_w_i * (score_gen_i - typicality_i_live)
                         + gen_w_j * (score_gen_j - typicality_j_live)).mean()
  ```
  where `typicality_*_live` is computed with grads enabled at this step.

Either way, the regularizer (KL to base or per-completion squared
penalty) should be added separately and gated by its own weight flag,
not bundled into the TC subtraction.

---

## See also

- `docs/IMPORTANT-RESEARCH-PLAN.md` §3(a): TC-as-reweighter analysis,
  live-with-grads proposal.
- `docs/IMPORTANT-RESEARCH-PLAN.md` §7: TODO entry for the KL/penalty
  regularizer.
- `docs/comb_loss_g_mode_concerns.md`: scope of `comb` loss in g-mode,
  including the locked design decisions for gen-NLL one-sidedness.
- `docs/issue3_fix.md`: per-pair vs per-item NLL gating, no `/2` factor.
