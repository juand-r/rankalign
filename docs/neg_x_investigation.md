# neg-x Prompting Investigation — Full Summary

**Question**: Can we use log P(y | neg-x prompt) to distinguish correct from incorrect
Python solutions? Specifically: does the model assign higher per-token probability to
wrong solutions when instructed to write incorrect code?

**Short answer**: No, not reliably. The approach fails in two distinct ways, and the
path to making it work has a fatal circular dependency.

---

## Background

The neg-x scoring idea: given a model M and two candidate solutions y_c (correct) and
y_w (wrong), compute:

$$s(y) = \frac{1}{|y|} \sum_{t} \log P(y_t \mid \text{prefix}_{neg\text{-}x}, y_{<t})$$

If the neg-x instruction genuinely shifts the model toward wrong code, we expect
$s(y_w) > s(y_c)$, i.e., AUROC < 0.5 when wrong is the "positive" class.

All results below use **gemma-4-31B-it** unless noted. "Lenorm" = mean per-token log P.
"Inversion" = AUROC(correct_label, neg_x_score) < 0.5 (higher probability to wrong solutions).
"Genuine inversion" = inversion that does not exist in the unconditional baseline (uncond AUROC ≥ 0.5).

---

## Part 1: Scoring Pre-Existing Wrong Solutions

### 1.1 Prompt Variant Sweep

8 prompt variants (V1–V3, VA–VF, ranging from generic to very specific instruction
wording) were tested on all 82 HumanEval tasks.

**Result**: Only 4/82 tasks invert on lenorm — and this is the same 4 tasks regardless
of which prompt variant is used:

| task_id | task | V1 lenorm AUROC |
|---------|------|-----------------|
| humaneval_26  | `remove_duplicates` | 0.450 |
| humaneval_149 | `sorted_list_sum`  | 0.467 |
| humaneval_115 | `max_fill`         | 0.382 |
| humaneval_35  | `max_element`      | 0.489 |

More forceful wording, technical framing, step-by-step instructions — none of it changes
which tasks invert or the inversion strength.

### 1.2 Length Artifacts

Raw sum log P inverts 8 tasks, but 6 of these are artifacts: correct solutions happen
to be longer, so their total log P is more negative. After per-token normalization, 4
remain. The uncond baseline (no instruction at all) already inverts these due to structural
differences in solution length — the neg-x instruction is not causing it.

### 1.3 Genuine vs Artifact Inversions

Testing: if uncond baseline (P(y | problem only)) also inverts, the effect is structural,
not instruction-driven.

| Task | Problem | uncond AUROC | neg_v1 AUROC | Verdict |
|------|---------|--------------|--------------|---------|
| humaneval_26 (`remove_duplicates`) | Keep elements appearing exactly once | 0.661 | 0.450 | **genuine** |
| humaneval_149 (`sorted_list_sum`) | Filter+sort strings by length | 0.724 | 0.467 | **genuine** |
| humaneval_115 (`max_fill`) | Water well bucket count | 0.406 | 0.382 | partial artifact |
| humaneval_35 (`max_element`) | Find list maximum | 0.452 | 0.489 | **pure artifact** |

**Net result: 2/82 tasks have genuine instruction-driven neg-x inversion.**

For humaneval_26, the wrong approach is `seen = set()` (keeps unique elements rather
than removing all duplicates). For humaneval_149, wrong solutions miss the filter or use
the wrong sort key. The neg-x instruction shifts probability toward these approaches.

### 1.4 Few-Shot Neg-X

Adding 1–2 in-context wrong-solution examples:

| shots | lenorm inverted | genuine |
|-------|----------------|---------|
| 0     | 4/82 | 2 |
| 1     | 3/82 | ~2 |
| 2     | 2/82 | ~2 |

**Few-shot is strictly worse.** The model is so strongly RLHF-trained toward correct code
that concrete examples of wrong solutions don't shift the distribution.

### 1.5 Base Models

Hypothesis: base models (no RLHF) don't resist the neg-x instruction and should show
more inversions.

Results across 3 base models (4 variants each: bare/0-shot, V1-text, 1-shot, 2-shot):

| Model | Best genuine inversions | Best variant |
|-------|------------------------|-------------|
| gemma-4-31B-it (RLHF) | **2/82** | V1 zero-shot |
| gemma-4-31B base | **2/82** | 0-shot / 1-shot / 2-shot |
| gemma-2-9b base | **2/82** | all variants |
| Llama-3.1-8B base | **3/82** | V1-text / few-shot |

Base models don't outperform the instruct model. They just invert *different* tasks
(humaneval_9 `rolling_max`, humaneval_130 `tri`) compared to the instruct model
(humaneval_26, humaneval_149). Llama reaches 3/82, the maximum achieved by any model
in any configuration.

**Conclusion for Part 1**: neg-x scoring, applied to pre-existing wrong solutions,
achieves at most 3/82 genuine inversions (3.7%). This is not useful as a general
scoring signal.

---

## Part 2: Why Scoring Fails — The Candidate Pool Problem

### 2.1 Generation Experiment

To understand *what* the neg-x prompt actually generates, we ran generation (sampling)
under 5 prompt variants, 10 HumanEval tasks, 5 samples each = **250 generated solutions**.

Fraction that fail the HumanEval test suite:

| Variant | Wrong (fail tests) | Description |
|---------|--------------------|-------------|
| `v1`               | **49/50 (98%)** | "intentionally incorrect Python" |
| `common_mistake`   | 30/50 (60%)     | "common mistakes beginners make" |
| `misconception`    | 16/50 (32%)     | "common conceptual misconception" |
| `confident_wrong`  | 11/50 (22%)     | "confidently but subtly wrong" |
| `edge_case`        | 7/50 (14%)      | "fails on edge cases only" |

The V1 prompt is extremely effective at *generating* wrong code — 98% failure rate.
But generation ≠ scoring.

### 2.2 Qualitative Analysis: What Makes Generated Wrong Solutions Different

V1-generated wrong solutions implement **coherent alternative algorithms**:

- `separate_paren_groups`: Tracks depth correctly but strips the parentheses from the
  output (`group.replace('(', '').replace(')', '')`) or returns `result[1:]`
- `rolling_max`: Returns the element itself instead of the running max, or appends
  `n` instead of `current_max`
- `remove_duplicates`: Uses `seen = set()` approach (removes all occurrences) instead
  of Counter-based filtering (keeps elements appearing exactly once)

Pre-existing wrong solutions in our evaluation set are a different population: some are
corrupted text, some have syntax errors, some are nearly correct with a single-token
slip. They are not drawn from the model's "wrong code" distribution.

### 2.3 Scoring Model-Generated Wrong Solutions

When we score the model's *own* V1-generated wrong solutions under the neg-x prompt
(instead of pre-existing wrong solutions), the result changes completely:

| Setup | Task-level inversions |
|-------|----------------------|
| Scoring pre-existing wrong solutions | 2–4 / 82 (3–5%) |
| Scoring model-generated wrong solutions | **10 / 10 (100%)** |

Per-task lenorm comparison (wrong_mean vs correct_mean under neg-x V1):

| Task | wrong_mean | correct_mean | inverted |
|------|-----------|--------------|---------|
| HumanEval/1   | -0.188 | -0.652 | ✓ |
| HumanEval/9   | -0.042 | -1.081 | ✓ |
| HumanEval/21  | -0.134 | -0.608 | ✓ |
| HumanEval/26  | -0.031 | -1.255 | ✓ |
| HumanEval/35  | -0.026 | -1.615 | ✓ |
| HumanEval/46  | -0.198 | -0.480 | ✓ |
| HumanEval/88  | -0.417 | -0.769 | ✓ |
| HumanEval/115 | -0.291 | -0.975 | ✓ |
| HumanEval/130 | -0.155 | -0.473 | ✓ |
| HumanEval/149 | -0.060 | -1.081 | ✓ |

The margin is large. Wrong solutions score ~0.5 log-prob-per-token higher on average.

**Why this works when pre-existing scoring doesn't**: The model's generated wrong
solutions are by definition in the support of $P(y \mid \text{neg-x})$. Scoring them
under that same distribution gives high probability. Pre-existing wrong solutions are
out-of-distribution for the neg-x prompt — the model has never learned to assign high
probability to that *style* of wrong code.

### 2.4 The Circular Dependency

This result doesn't help us. To score candidate solutions under neg-x:

1. If the wrong candidates come from the neg-x distribution → the scoring works (10/10)
2. If the wrong candidates come from any other source → the scoring mostly fails (2–4/82)

But the *goal* of neg-x scoring was to rank arbitrary candidate solutions — not just the
ones the model generated. If we already have model-generated wrong solutions, we can
identify them as wrong just by running the test suite. We don't need scoring.

---

## Part 3: AST Mutation — A Model-Free Alternative

### 3.1 Motivation

The generation experiment showed the problem: candidates must come from the neg-x
distribution for scoring to work, but that's circular. What if we could create wrong
solutions without using the model at all?

Rule-based AST mutation: parse the canonical correct solution, apply a deterministic
transformation, verify it fails the test suite. The wrong solution is model-free —
it doesn't come from any learned distribution.

### 3.2 Mutation Types

Five AST `NodeTransformer` classes applied to `canonical_solution`:

| Mutator | Transformation | Example |
|---------|---------------|---------|
| `compare_flip` | Flip all comparison operators: `>↔<`, `==↔!=`, etc. | `if n > max_val` → `if n < max_val` |
| `denom_plus_one` | Add 1 to all division denominators | `x / n` → `x / (n + 1)` |
| `return_slice` | Truncate every non-trivial return value | `return result` → `return result[:-1]` |
| `range_minus_one` | Subtract 1 from all `range()` upper bounds | `range(n)` → `range(n - 1)` |
| `append_loop_var` | In tracking loops, append loop variable instead of accumulator | `result.append(current_max)` → `result.append(n)` |

Tested on 10 tasks × 5 mutators = 50 (task, mutator) pairs.

**Test outcomes:**

| Mutator | fail/total | fail% | Notes |
|---------|-----------|-------|-------|
| `compare_flip`  | 7/10 | 70% | Strongest signal — flipped condition logic |
| `return_slice`  | 8/10 | 80% | Works broadly; trivially wrong off-by-one |
| `denom_plus_one`| 5/10 | 50% | Only works in tasks with division |
| `range_minus_one`| 4/10 | 40% | Misses by 1 iteration |
| `append_loop_var`| 1/10 | 10% | Very specific pattern |

18 plausible failing mutants (AssertionError on test suite), 5 syntactic crashes, 27 pass.
The 27 passers are tasks where the mutation is semantically neutral for that problem.

### 3.3 Scoring Results

All 18 plausible failing mutants + 50 correct solutions scored under neg-x V1 prompt.

**Per-pair inversions**: 17/18 (94%)
**Task-level inversions**: 10/10 (100%)

Per-task detail:

| Task | mutant lenorm | correct lenorm | inverted | mutators |
|------|--------------|---------------|---------|---------|
| HumanEval/1   | -0.647 | -0.652 | ✓ (narrow) | `return_slice` |
| HumanEval/9   | -0.523 | -1.081 | ✓ | `append_loop_var`, `return_slice` |
| HumanEval/21  | -0.447 | -0.608 | ✓ | `return_slice`, `denom_plus_one` |
| HumanEval/26  | -0.864 | -1.255 | ✓ | `return_slice`, `compare_flip` |
| HumanEval/35  | -0.290 | -1.615 | ✓ | `compare_flip` |
| HumanEval/46  | -0.430 | -0.480 | ✓ | `range_minus_one` |
| HumanEval/88  | -0.377 | -0.769 | ✓ | `return_slice` |
| HumanEval/115 | -0.308 | -0.975 | ✓ | `denom_plus_one` |
| HumanEval/130 | -0.405 | -0.473 | ✓ | all 4 working mutators |
| HumanEval/149 | -0.327 | -1.081 | ✓ | `return_slice`, `compare_flip` |

The one failure: `compare_flip` on HumanEval/1. This task's correct solutions happen to
score unusually high under neg-x (mean -0.652), and the compare_flip mutant scores -0.763
(slightly lower than correct, so no inversion).

The strongest inversions — tasks 35, 115, 149 — have margins of 0.67–1.32 log-prob/token.

### 3.4 Why AST Mutations Work

The model assigns high probability to structurally plausible code. A mutant produced by
`compare_flip` is syntactically identical to correct code (same structure, same identifiers,
only operator tokens differ). Under the neg-x prompt, the model prefers plausible-looking
wrong code over correct code — and AST mutations are exactly that.

This is the same mechanism as model-generated wrong solutions, but without the circularity.
The mutants don't come from the model's distribution, yet they still sit in the region of
token-space the model prefers under neg-x.

---

## Summary and Open Questions

### What we learned

| Finding | Result |
|---------|--------|
| neg-x prompt variants (8 tested) | Same 4 tasks invert regardless of wording |
| Genuine inversions (pre-existing wrong solutions) | 2/82 tasks (instruct), 2–3/82 (base) |
| Few-shot neg-x | Worse than zero-shot |
| Base models vs instruct | No advantage; different tasks, same rate |
| Scoring model-generated wrong solutions | 10/10 tasks invert — but circular |
| AST mutations (model-free) | 17/18 (94%) per-pair inversions, 10/10 task-level |

### Key insight

The neg-x scoring signal is real but narrow. It only works when:
1. The wrong candidate looks plausible (similar structure to correct code), AND
2. The wrong candidate is in the neg-x distribution (or close to it).

Pre-existing wrong solutions in evaluation datasets don't satisfy (2). Model-generated
wrong solutions satisfy (2) by construction but require running the model (circular).
AST mutations satisfy both without any model dependency.

### Open questions for deciding where to go next

1. **Scale**: The mutation experiment covers 10 tasks × 18 mutants. Does the 94% inversion
   rate hold across all 82 HumanEval tasks? A broader sweep would tell us if this is
   a general phenomenon or a property of these specific tasks/mutators.

2. **Mutation coverage**: Only tasks where a mutation is semantically meaningful produce
   failing mutants. For example, `append_loop_var` only matches 1/10 tasks. Designing
   more mutators (wrong initialization, wrong condition in filter, off-by-one in slice
   index) could increase coverage. Alternatively, using LLM-assisted mutations constrained
   to preserve syntax structure.

3. **What is the neg-x signal actually measuring?** The model assigns high probability to
   wrong-but-plausible code under neg-x. But does this reflect a meaningful property of
   the code, or just surface features (short length, common patterns)? A study that holds
   surface features constant while varying semantic correctness would clarify this.

4. **Can this build a useful scoring metric?** If AST mutations give reliable neg-x
   inversions, one application is: given an unknown solution y, create AST-mutated
   versions y_1…y_k, and check if neg-x ranks the mutants above the original. If it does,
   the original is "correct-looking." If not, the original itself may be in the wrong
   distribution. This would be a no-test-suite correctness proxy.

5. **Comparison to CPMI**: The CPMI metric (conditional pointwise mutual information,
   explored in v2.1 analysis) approaches the same question differently. How does neg-x
   AUROC compare to CPMI on the same task set?
