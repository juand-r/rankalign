# Testing the Quality Constraint Empirically

## Setup

Given a question Q and candidate answer a:

- **G(a)** = P_LLM(a | positive prompt) — generator probability
- **G'(a)** = P_LLM(a | negative prompt) — negative generator probability
- **V(a)** = P_LLM(Yes | validator prompt for a) — validator probability

The **quality constraint** says: the generator never produces an answer it believes is incorrect. Formally, π(a|c,Q) = 0 whenever c_a = 0, where c is the configuration of which answers are correct and π is the generation policy.

## Violation test: G(a) > V(a)

**Result**: G(a) ≤ V(a) is a necessary condition for the quality constraint. Equivalently, G(a) > V(a) is a definitive proof of violation. This holds at any level of epistemic uncertainty.

**Proof**: Under the quality constraint, all generation mass on a comes from configurations where c_a = 1:

    G(a) = Σ_{c: c_a=1} P(c|Q) π(a|c,Q) ≤ Σ_{c: c_a=1} P(c|Q) = V(a)

since π(a|c,Q) ≤ 1. If G(a) > V(a), the surplus G(a) - V(a) is a lower bound on the violation mass (generation from c_a = 0 configurations).

**Limitation**: G(a) ≤ V(a) is not sufficient. Violation mass from c_a = 0 configurations can be present yet small enough that G(a) still falls below V(a). This happens because we only observe total G(a), not its decomposition into legitimate mass (from c_a = 1 configs) and violation mass (from c_a = 0 configs). When 0 < V(a) < 1, the legitimate mass can be strictly less than V(a), leaving room for violation mass to hide in the gap.

## Stronger interpretation at V(a) ≈ 0 (zero-epistemic-uncertainty subset)

**Definition**: A subset S of answers has *zero epistemic uncertainty* if V(a) ∈ {0,1} for all a ∈ S. This means the marginal distribution of (C_a)_{a ∈ S} is a point mass: for every a ∈ S, C_a is determined with probability 1 — equal to 1 if V(a) = 1 and 0 if V(a) = 0.

(Proof that V(a) ∈ {0,1} for all a implies the marginal is a point mass: P(C_a ≠ c\*_a | Q) = 0 for each a ∈ S, so by the union bound P(∃ a ∈ S: C_a ≠ c\*_a | Q) ≤ Σ_{a ∈ S} P(C_a ≠ c\*_a | Q) = 0.)

**Result**: On a subset S with zero epistemic uncertainty, G(a) ≤ V(a) for all a ∈ S is both necessary and sufficient for the quality constraint restricted to S.

**Proof of sufficiency**: Take any a ∈ S.

- If V(a) = 1: P(c_a = 0 | Q) = 0, so there are no configurations with c_a = 0 and positive probability. The quality constraint at a is vacuously satisfied.

- If V(a) = 0: P(c_a = 1 | Q) = 0, so P(c|Q) = 0 for all c with c_a = 1. Therefore the first sum in G(a) vanishes:

      G(a) = 0 + Σ_{c: c_a=0} P(c|Q) π(a|c,Q)

  We assumed G(a) ≤ V(a) = 0, and G(a) ≥ 0, so G(a) = 0. Therefore Σ_{c: c_a=0} P(c|Q) π(a|c,Q) = 0. Since each term P(c|Q) π(a|c,Q) ≥ 0 and they sum to zero, every term is zero. In particular, for any c with c_a = 0 and P(c|Q) > 0, we have π(a|c,Q) = 0 — which is exactly the quality constraint at a.

**Why sufficiency fails generally but holds here**: The hiding argument requires 0 < V(a) < 1 so that both the legitimate sum (c_a = 1) and violation sum (c_a = 0) are potentially positive. When V(a) = 0, the legitimate sum is killed, so G(a) equals the violation mass exactly — there is nothing to hide behind. When V(a) = 1, the quality constraint is vacuously satisfied.

**Caveat**: Passing the test on S does not guarantee the quality constraint holds for answers outside S.

## Joint distribution coherence testing

We test whether the model behaves as if it has a coherent joint distribution over answer correctness. The test domain is noble gases with 18 targets (7 noble gases, 5 other gases, 1 tricky name, 1 other element, 1 non-element, 1 wrong category, 1 absurd, 1 misspelling). We probe all 306 ordered pairs (a, b) with a ≠ b.

Scripts:
- `probe_noble_joint.py` — few-shot probes (initial version)
- `probe_conditional_ablation.py` — ablation on positive conditional prompts (4 phrasings × 3 few-shot)
- `probe_neg_conditional_ablation.py` — ablation on negative conditional prompts (4 phrasings × 4 few-shot)
- `probe_pos_suppose_test.py` — test "Suppose" vs declarative for positive conditional
- `probe_noble_joint_zeroshot.py` — zero-shot declarative probes (superseded)
- `probe_noble_joint_suppose.py` — zero-shot "Suppose" probes (**current best**, Hilium excluded)
- `analyze_noble_joint.py` — coherence analysis for declarative probes
- `analyze_noble_joint_suppose.py` — coherence analysis for "Suppose" probes
- `probe_triple_conditional_ablation.py` — triple-conditional phrasing ablation (conjunctive/sequential/comma)

### Probe types and prompt templates

Four probe types, each eliciting P(Yes) as a proxy for the target probability:

**Marginal P(Ca=1):**
- Few-shot: 2 examples (Jupiter/rainbow→No, Mars/planet→Yes), then `Q: Is "{a}" a noble gas? Answer Yes or No.`
- Zero-shot: `Q: Is "{a}" a noble gas? Answer Yes or No.\nA:`

**Conditional-positive P(Ca=1 | Cb=1):**
- Few-shot: 2 examples (Jupiter planet / Blue planet → No; Red rainbow / Blue rainbow → Yes), then `Q: "{b}" is a noble gas. Is "{a}" also a noble gas?`
- Zero-shot: `Q: "{b}" is a noble gas. Is "{a}" a noble gas? Answer Yes or No.\nA:`
- **Ablation** (positive only): tested 4 phrasings × 3 few-shot settings = 12 variants (`probe_conditional_ablation.py`). Phrasings: "also", "no_also" (drop "also"), "assume_independently", "what_if". Few-shot settings: zero-shot, 2-example (current), 4-example (chemistry domain). Finding: **few-shot examples cause severe yes-bias in 2b-it** regardless of phrasing (e.g., P(Hydrogen|Neon) jumps from 0.005 zero-shot to 0.997 few-shot). 9b-it is robust. Zero-shot "no_also" phrasing chosen for final probes.

**Conditional-negative P(Ca=1 | Cb=0):**
- Few-shot: 2 examples (Jupiter NOT rainbow / Blue rainbow → Yes; Moon NOT planet / Mars planet → Yes), then `Q: "{b}" is NOT a noble gas. Is "{a}" a noble gas?`
- Zero-shot declarative: `Q: "{b}" is NOT a noble gas. Is "{a}" a noble gas? Answer Yes or No.\nA:`
- **Ablation** (`probe_neg_conditional_ablation.py`): tested 4 phrasings × 4 few-shot settings = 16 variants. Phrasings: "declarative_not" (`"{b}" is NOT...`), "suppose_not" (`Suppose "{b}" is not...`), "if_not" (`If "{b}" were not...`), "given_not" (`Given that "{b}" is not...`). Few-shot settings: zero-shot, current 2-example (both Yes — biased), balanced 2-example (one Yes, one No), chemistry 4-example.
- **Finding:** The declarative zero-shot prompt was severely broken for 2b-it. P(Helium | Carbon is NOT noble) collapsed to 0.10 (should be ~0.99). P(Krypton | Methane is NOT noble) collapsed to 0.02. The all-caps "NOT" hijacks the small model into rejecting everything. Ranked by coherence RMSE on the negative joint: `suppose_not__zero_shot` (0.134) best zero-shot; `given_not__balanced_2ex` (0.114) best overall. `declarative_not__zero_shot` (0.465) second-worst. `if_not__zero_shot` (0.490) worst.

**Joint P(Ca=1, Cb=1):**
- Few-shot: 2 examples (Blue+Jupiter rainbow → No; Mars+Jupiter planet → Yes), then `Q: Are "{a}" and "{b}" both noble gases?`
- Zero-shot: `Q: Are "{a}" and "{b}" both noble gases? Answer Yes or No.\nA:`
- Not ablated.

### Coherence checks performed

**Positive joint** (P(Ca=1, Cb=1)):
- P(Ca=1|Cb=1) · P(Cb=1) vs P(Cb=1|Ca=1) · P(Ca=1) — two factorizations of the joint
- Also compared against directly probed P(Ca=1, Cb=1) from joint prompt

**Negative joint** (P(Ca=1, Cb=0)):
- P(Ca=1|Cb=0) · (1−P(Cb)) vs (1−P(Cb|Ca=1)) · P(Ca) — two factorizations
- Total probability law: P(Ca) vs P(Ca|Cb=1)·P(Cb) + P(Ca|Cb=0)·(1−P(Cb))
- These are theoretically equivalent given positive-case coherence, confirmed empirically (nearly identical RMSE).

### Prompt sensitivity

Few-shot examples cause severe yes-bias in smaller models (gemma-2-2b-it): P(Hydrogen is noble | Neon is noble) jumps from 0.005 (zero-shot) to 0.997 (few-shot), regardless of phrasing variant. Larger models (gemma-2-9b-it) are robust to this. **Zero-shot prompts are required for reliable conditional probing in smaller models.**

### Unified "Suppose" framing

After ablating both positive and negative conditionals separately, we chose a unified "Suppose" framing for consistency — needed for compound conditionals like P(a=1 | b=1, c=0, d=1).

Final prompt templates (`probe_noble_joint_suppose.py`):
- **Marginal:** `Q: Is "{a}" a noble gas? Answer Yes or No.` (unchanged)
- **Cond-positive:** `Q: Suppose "{b}" is a noble gas. Is "{a}" a noble gas? Answer Yes or No.`
- **Cond-negative:** `Q: Suppose "{b}" is not a noble gas. Is "{a}" a noble gas? Answer Yes or No.`
- **Joint:** `Q: Are "{a}" and "{b}" both noble gases? Answer Yes or No.` (unchanged)

Trade-off for 2b-it: "Suppose" slightly worse than declarative for positive case (RMSE 0.025 vs 0.009 on ablation test pairs), but dramatically better for negative case (RMSE 0.134 vs 0.465). For 9b-it, essentially identical on both.

Coherence results with "Suppose" prompts (Hilium excluded, 17 targets, 272 pairs):

                                    ── 2b-it ──           ── 9b-it ──
    Test                         RMSE   corr   max|Δ|   RMSE   corr   max|Δ|
    Pos: P(a|b)P(b) vs P(b|a)P(a)  0.155  0.881  0.604   0.175  0.890  0.974
    Neg: P(a|b=0)(1-Pb) vs         0.234  0.828  0.879   0.207  0.894  0.982
         (1-P(b|a))P(a)
    Total probability law           0.231  0.868  0.879   0.208  0.915  0.982
    Factored vs direct joint        0.430  0.514  0.994   0.207  0.866  0.995

Compared to the old declarative zero-shot (on same 17 targets):

                                    ── 2b-it ──           ── 9b-it ──
    Test                         RMSE   corr            RMSE   corr
    Pos: P(a|b)P(b) vs P(b|a)P(a)  0.166  0.882           0.209  0.847
    Neg: P(a|b=0)(1-Pb) vs         0.446  0.459           0.217  0.884
         (1-P(b|a))P(a)
    Total probability law           0.445  0.594           0.217  0.908

Main improvement: 2b-it negative RMSE halved (0.446→0.234), correlation nearly doubled (0.459→0.828).

Remaining top discrepancies: 9b-it errors are almost all Nobelium (model incorrectly classifies as noble). 2b-it errors dominated by Oxygen (model ~50/50 on Oxygen being noble) and Nobelium.

### Pragmatic context effects (fundamental limitation)

Some apparent incoherence is actually pragmatic disambiguation, not Bayesian inconsistency. For example, the model assigns P(Hilium is noble) ≈ 0.93 in isolation (interpreting the misspelling as "Helium"), but P(Hilium is noble | Helium is noble) ≈ 0.006 — seeing the correct spelling next to the misspelling forces the model to treat them as distinct entities. A human would do the same: the identity of the referent changes with context.

Similarly, P(Nobelium is noble) ≈ 0.96 in isolation (fooled by the name), but P(Nobelium | Argon is noble) ≈ 0.11 — seeing a real noble gas for comparison triggers reconsideration.

This is not a failure of Bayesian updating. The propositions being evaluated are not the same across contexts — the meaning of "Hilium" or "Nobelium" shifts when other elements are mentioned. This is a fundamental limitation of probing LLMs with natural language: the "same question" can denote different events in different contexts, so checking P(A|B)P(B) = P(B|A)P(A) is not a clean test of probabilistic coherence when A and B are natural-language propositions whose interpretation is context-dependent.

Hilium is excluded from quantitative coherence metrics going forward for this reason (it's a misspelling, not a real element). Nobelium is kept as a legitimate knowledge test.

### Triple-conditional ablation

Before scaling to full configuration elicitation (for estimating r(a)), we tested whether conditioning on two premises at once (P(a=1 | b=±, c=±)) gives coherent 3-way joints via different factorizations.

Script: `probe_triple_conditional_ablation.py`

**Targets:** 8 elements (Helium, Neon, Argon, Krypton, Hydrogen, Oxygen, Carbon, Nobelium)

**8 diagnostic triples:** (He,Ar,Ne), (He,Ar,H), (He,H,O), (He,Ar,No), (H,O,C), (Ar,Kr,Ne), (Ne,H,C), (Kr,O,No)

**Phrasing variants tested (all zero-shot):**
1. **Conjunctive:** `Q: Suppose "B" is [not] a noble gas and "C" is [not] a noble gas. Is "A" a noble gas? Answer Yes or No.`
2. **Sequential:** `Q: Suppose "B" is [not] a noble gas. Suppose "C" is [not] a noble gas. Is "A" a noble gas? Answer Yes or No.`
3. **Comma:** `Q: Suppose "B" is [not] a noble gas, and "C" is [not] a noble gas. Is "A" a noble gas? Answer Yes or No.`

**Consistency check:** For each triple (a,b,c) and phrasing, compute 6 factorizations of P(a=1,b=1,c=1):
- F1: P(a|b=1,c=1) · P(b|c=1) · P(c)
- F2: P(a|b=1,c=1) · P(c|b=1) · P(b)
- F3: P(b|a=1,c=1) · P(a|c=1) · P(c)
- F4: P(b|a=1,c=1) · P(c|a=1) · P(a)
- F5: P(c|a=1,b=1) · P(a|b=1) · P(b)
- F6: P(c|a=1,b=1) · P(b|a=1) · P(a)

Measure: spread = max(F1..F6) - min(F1..F6). Smaller = more coherent.

**Results — 9b-it:** Phrasing barely matters. Spread is tiny for noble gas triples (0.001–0.003 for (He,Ar,Ne)), moderate for triples with Krypton (0.03), and large only for Nobelium triples (0.97). All three phrasings give nearly identical results.

**Results — 2b-it:** Phrasing makes a significant difference.
- (He,Ar,Ne): conjunctive spread = 0.41, sequential spread = **0.08**, comma spread = 0.25
- Sequential phrasing ("Suppose X. Suppose Y.") preserves marginals much better for the smaller model.
- Other triples show a similar pattern: sequential is consistently tightest.

**Conclusion:** Use **sequential** phrasing ("Suppose X. Suppose Y.") for multi-premise conditioning. Best for 2b-it and comparable for 9b-it.

---

## Generator, validator, and negative generator prompt selection

The G-V gap analysis uses three prompt types with a fixed configuration chosen through systematic comparison across multiple variants. All probing was done with `quick_probe_noble_fewshot.py` on both gemma-2-2b-it and gemma-2-9b-it.

### Generator s_G: `instructed-zero`

**Prompt:**
```
Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else.

Q: Name a noble gas.
A:
```

**Selection rationale:** Tested 7 generator prompts (zero-shot, few-shot, few-shot-v2, few-shot-chem, instructed-zero, instructed-few, instructed-chem). `instructed-zero` was chosen because it gives the most balanced spread across correct noble gases (42% Helium, 26% Neon, 29% Argon for 9b-it) with 98.9% of mass on noble gases and zero formatting junk. Other prompts were Helium-dominated (93%+ on Helium alone). The initial `instructed-zero` attempt used "capitalized" phrasing, which the model interpreted as ALL CAPS; fixed by replacing with explicit examples `(e.g. "Sodium", "Chlorine")`.

### Negative generator s_G': `neg-instructed-zero`

**Prompt:**
```
Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else.

Q: Name something that is NOT a noble gas.
A:
```

**Selection rationale:** Tested 5 negative prompts (neg-zero-shot, neg-instructed-zero, neg-few-shot, neg-few-shot-v2, neg-instructed-few). `neg-instructed-zero` chosen as the natural match to `instructed-zero` — same style, just negated. For 9b-it only 3.5% of mass on noble gases (good). For 2b-it 20.5% (noisier but best among options).

### Validator s_V: `val-few-shot`

**Prompt:**
```
Q: Is "Jupiter" a color of the rainbow? Answer Yes or No.
A: No

Q: Is "Mars" a planet in the solar system? Answer Yes or No.
A: Yes

Q: Is "{X}" a noble gas? Answer Yes or No.
A:
```

**Selection rationale:** Tested 3 validator prompts (val-zero, val-instructed, val-few-shot). `val-few-shot` chosen for highest Yes+No mass coverage and cleanest discrimination. For 9b-it: noble gas logodds 6.8–9.7, non-noble below -1.7. For 2b-it: generally good but incorrectly gives positive logodds to Fluorine (4.61), Sodium (4.01), Oxygen (3.19) — wrong beliefs, not validator noise.

### Score types

- **s_G, s_G'** are **log-probabilities**: log P(completion | prompt), summed over tokens via `F.log_softmax`
- **s_V** is **log-odds**: log P(Yes) - log P(No), where P(Yes) and P(No) are each summed over 5 token variants (" Yes", "Yes", " yes", "yes", " YES" and same for No) via `logsumexp`

---

## Practical considerations for LLMs

### Thresholds

LLM probabilities come from softmax, which always produces values in (0, 1) — never exactly 0 or 1. We need tolerance thresholds to apply the theory.

**noise_thresh** ≈ 0.001. We require G(a) - V(a) > noise_thresh to flag a violation. This single threshold means: differences below noise_thresh are not trusted, whether because both values are tiny (absolute noise) or because the surplus is negligible at larger magnitudes (bfloat16 has ~0.8% relative precision, so differences < 0.001 at any magnitude could be rounding). At worst this hides a violation of mass < 0.001, which is not practically interesting. Also used to grey out cases where both G and V are below noise_thresh.

**ε_val** = max(0.001, 1 - avg_yes_no_mass). This is the validator certainty threshold, below which V(a) is treated as "near zero." It is used only for *interpretation* (not for violation detection): when V(a) < ε_val and G(a) ≤ V(a), the quality constraint is **confirmed** at that answer — not just inconclusive — because nearly all generation mass would be violation mass if present. It accounts for:
- **Yes/No mass leakage**: Even when the model is maximally certain, some probability mass lands on tokens other than Yes/No. If the average Yes+No mass is 0.998, a V(a) of 0.002 may represent maximal certainty that a is wrong, not a 0.2% credence. Setting ε_val ≥ 1 - avg_yes_no_mass ensures we don't misread leakage as uncertainty.
- **bfloat16 precision floor**: The max with 0.001 ensures a minimum tolerance even if Yes+No mass happens to be very high.

### Classification of answers

For each candidate answer a, using the single test G(a) > V(a):

1. V(a) ≈ 1: **vacuous** — quality constraint can't be violated
2. G(a) - V(a) > noise_thresh: **violation** — definitive proof of quality constraint violation, with meaningful surplus mass
3. Both G(a) and V(a) < noise_thresh: **noise** — comparison unreliable, mass negligible either way
4. G(a) ≤ V(a) and V(a) < ε_val: **cleared** — quality constraint confirmed at this answer (V ≈ 0 means G ≈ violation mass, and it's ≤ V ≈ 0)
5. Otherwise: **inconclusive** — necessary condition G ≤ V holds but violations could be hiding

---

## Estimating r(a) structurally (planned)

The key theoretical relationship is:

    s_V(a) = s_G(a) - s_G'(a) + log r(a,Q)

where r(a,Q) = E[π'(a|c,Q) | C_a=0, Q] / E[π(a|c,Q) | C_a=1, Q].

The goal is to estimate r(a) from the model's own beliefs, then compare log r against the directly observed s_V - (s_G - s_G').

### Estimation procedure

1. **Fix candidate universe** A = {Helium, Neon, Argon, Krypton, Xenon, Radon, Hydrogen, Oxygen, Carbon, ...}
2. **Fix focal answer** a (e.g., Helium)
3. **Enumerate configurations** c ∈ {0,1}^|A|. For denominator: configs with c_a=1. For numerator: configs with c_a=0.
4. **Estimate P(c|C_a=1,Q)** via sequential yes/no elicitation in random orderings, then average over K orderings and renormalize.
5. **Estimate π(a|c,Q)** — generation probability conditioned on configuration.
6. **Estimate π'(a|c,Q)** — negative generation probability conditioned on configuration.
7. **Compute** D(a) = Σ_{c:c_a=1} P(c|C_a=1,Q) π(a|c,Q) and N(a) = Σ_{c:c_a=0} P(c|C_a=0,Q) π'(a|c,Q).
8. **r(a) = N(a)/D(a)**.

### Prompt design for π and π'

**Important distinction:** The "Suppose" sequential phrasing was validated for *conditional yes/no probing* (configuration elicitation). Estimating π and π' is a different task — it is a *generation* task where we score candidate completions, not Yes/No tokens. There is no reason to use the same prompt format for both.

For π and π', the prompt should:
1. State the **full configuration** explicitly — both correct and incorrect candidates. Listing only the correct set leaves the status of other candidates ambiguous, which is not what we want since a configuration is a complete binary vector.
2. Use the **same configuration prefix** for both π and π'. The only difference should be the final generation request (positive vs negative).
3. Keep the generation request close to the validated `instructed-zero` / `neg-instructed-zero` prompts.

**Proposed format** (for config where correct = {Helium, Neon, Argon, Xenon}, incorrect = {Hydrogen, Oxygen, Carbon}):

Configuration prefix (shared by π and π'):
```
Consider only the following candidates: Helium, Neon, Argon, Xenon, Hydrogen, Oxygen, Carbon.
The correct answers are exactly: Helium, Neon, Argon, Xenon.
The incorrect answers are exactly: Hydrogen, Oxygen, Carbon.
```

**π(a|c,Q) prompt** — append the positive generator request:
```
[config prefix above]

Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else.

Q: Name a noble gas.
A:
```

**π'(a|c,Q) prompt** — append the negative generator request:
```
[config prefix above]

Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else.

Q: Name something that is NOT a noble gas.
A:
```

Then score all candidate tokens and normalize: π over the correct set S+(c), π' over the incorrect set S−(c).

### π/π' ablation experiments

**Sanity check** (`probe_pi_sanity_check.py`): Tested the original declarative format ("The correct answers are exactly: ...") with Config A (Argon correct) vs Config B (Argon incorrect). Result: the declarative format **failed** — 9b-it showed no shift in Argon's probability, and 2b-it shifted in the wrong direction.

**Prompt variant ablation** (`probe_pi_ablation.py`): Tested 4 variants:
- `declarative`: original "The correct/incorrect answers are exactly: ..." prefix
- `suppose`: sequential Suppose premises (matching config elicitation style)
- `hypothetical`: "In a hypothetical scenario where..." framing
- `suppose_pool`: sequential Suppose premises + explicit pool restriction ("Your answer MUST be one of: ...")

Result: **`suppose_pool` was the clear winner** — the only variant that reliably conditioned both π and π' for both models. The pool restriction is essential for forcing the model to generate from the intended candidate set.

**Chosen format for π/π'** — `suppose_pool`:
```
Suppose "Hydrogen" is not a noble gas. Suppose "Oxygen" is not a noble gas. Suppose "Carbon" is not a noble gas. Suppose "Helium" is a noble gas. Suppose "Neon" is a noble gas. Suppose "Argon" is a noble gas. Suppose "Xenon" is a noble gas.

Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: Helium, Neon, Argon, Xenon.

Q: Name a noble gas.
A:
```
(π' uses the negative generation request and the incorrect-set pool instead.)

**Random configs test** (`probe_pi_random_configs.py`): Verified `suppose_pool` across 26 diverse correct/incorrect splits. Conditioning worked reliably — candidates received higher π mass when listed as correct and higher π' mass when listed as incorrect.

### Order sensitivity experiments

**Order sensitivity test** (`probe_pi_order_sensitivity.py`): Tested grouped (correct-first then incorrect) and interleaved (all shuffled together) permutations of Suppose premises. Found **severe order sensitivity** for both models: probability ranges of 0.5–0.9 for individual candidates across different orderings. Interleaving helped 9b-it π but hurt 2b-it π.

**Group order test** (`probe_pi_group_order.py`): Tested correct-first vs incorrect-first premise grouping. Key finding: **whichever group comes LAST (closest to the question) has the strongest influence** — a clear recency bias. This means group ordering is another source of variability requiring averaging.

**Recency bias mitigation test** (`probe_pi_recency_mitigation.py`): Tested 3 strategies to reduce order sensitivity without averaging:
1. `repeat_pool`: Add "Remember: the noble gases are exactly: ..." between premises and question
2. `numbered`: Use a numbered list format instead of sequential Suppose
3. `chat_sep`: Put configuration in a separate first chat turn, question in second turn

All used task-appropriate grouped ordering (incorrect-first/correct-last for π, and vice versa for π').

Results (max range across 5 within-group permutations):

| Model | Task | baseline | repeat_pool | numbered | chat_sep |
|-------|------|----------|-------------|----------|----------|
| 2b-it | π (4c_3i) | 0.307 | **0.136** | 0.363 | 0.708 |
| 2b-it | π' (4c_3i) | 0.591 | 0.367 | **0.296** | **0.275** |
| 9b-it | π (4c_3i) | **0.059** | 0.619 | 0.052 | 0.705 |
| 9b-it | π' (4c_3i) | 0.509 | **0.418** | 0.618 | 0.579 |

**Conclusion: No single prompt-level mitigation reliably reduces order sensitivity.** `repeat_pool` helps 2b π but hurts 9b π. `chat_sep` is the most volatile. `numbered` is inconsistent. **Order averaging (across multiple random permutations including group-order variation) remains the only robust strategy.**

### Final settled prompts for π/π'

**Format:** `suppose_pool` — sequential Suppose premises + pool restriction.

**π(a|c,Q) template:**
```
Suppose "{x1}" is [not] a noble gas. Suppose "{x2}" is [not] a noble gas. ... Suppose "{xN}" is [not] a noble gas.

Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: {correct_set}.

Q: Name a noble gas.
A:
```

**π'(a|c,Q) template:**
```
Suppose "{x1}" is [not] a noble gas. Suppose "{x2}" is [not] a noble gas. ... Suppose "{xN}" is [not] a noble gas.

Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: {incorrect_set}.

Q: Name something that is NOT a noble gas.
A:
```

**Premise ordering convention:**
- For π: incorrect premises first, correct premises last (so recency bias aligns with the generation target)
- For π': correct premises first, incorrect premises last (same logic — target group last)
- Within each group, order is shuffled randomly per evaluation

**Order averaging (required):** Each evaluation of π(a|c,Q) or π'(a|c,Q) must average over K random premise orderings to produce a stable estimate. The set of orderings should include:
- Both group orders (correct-first and incorrect-first) to cancel systematic recency bias
- Random within-group shuffles
- K to be determined by convergence test (proposed: K=20–30)

**Scoring:** Score all candidates via log P(completion | prompt), then pool-normalize over the task-appropriate subset (correct set for π, incorrect set for π'). Use logsumexp for numerical stability.

### Configuration elicitation prompt

Uses the validated sequential "Suppose" phrasing for each step:
```
Q: Suppose "Helium" is a noble gas. Suppose "Neon" is not a noble gas. Suppose "Argon" is not a noble gas. Is "Xenon" a noble gas? Answer Yes or No.
A:
```

### Status

- Configuration elicitation (P(c|Q)): pairwise validated, triple validated, sequential "Suppose" phrasing chosen
- π/π' prompts: `suppose_pool` format validated through sanity check, ablation, random configs, order sensitivity, group order, and recency mitigation tests
- Order sensitivity: confirmed severe; prompt-level mitigations (repeat-pool, numbered list, chat separation) all insufficient; **order averaging is the only robust strategy**
- Next step: convergence test — how many orderings K are needed for the order-averaged estimate to stabilize?
- Full r(a) estimation: not yet run. Plan is to start with a pilot on 1 question, ~9 candidates, 2 focal answers, K=20–30 orderings, using order-averaged `suppose_pool` for π/π'
