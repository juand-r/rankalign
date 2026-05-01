# Plan: Structural Estimation of log r(a, Q)

## Theoretical Background

From the formal derivation (Eq. 6), the relationship between generator and validator scores is:

$$s_V(a) = s_G(a) - s_{G'}(a) + \log r(a, Q)$$

where:
- $s_V(a)$ is the **validator log-odds**: $\log P(\text{Yes} \mid a) - \log P(\text{No} \mid a)$
- $s_G(a)$ is the **generator log-probability**: $\log P(a \mid \text{positive prompt})$
- $s_{G'}(a)$ is the **negative generator log-probability**: $\log P(a \mid \text{negative prompt})$
- $r(a, Q)$ is the **policy ratio**:

$$r(a, Q) = \frac{E[\pi'(a \mid \mathbf{c}, Q) \mid C_a = 0, Q]}{E[\pi(a \mid \mathbf{c}, Q) \mid C_a = 1, Q]}$$

Here $\mathbf{c} \in \{0,1\}^{|A|}$ is a configuration (which candidates are correct), $\pi(a \mid \mathbf{c}, Q)$ is the probability of generating $a$ under the positive prompt given configuration $\mathbf{c}$, and $\pi'(a \mid \mathbf{c}, Q)$ is the same under the negative prompt.

**What $r$ measures:** The ratio of "how much negative-generation budget $a$ gets when it's incorrect" to "how much positive-generation budget $a$ gets when it's correct." If this ratio is roughly constant across candidates, then the neg-typicality corrected score $s_G - s_{G'}$ should correlate perfectly with the validator score $s_V$.

**What we already know:** We can compute $\log r$ directly from observed scores as $s_V - (s_G - s_{G'})$. Prior analysis showed this is NOT constant (std ~5–8, strong anti-correlation with corrected score). The goal now is to **structurally estimate** $r(a)$ by decomposing it into its constituent parts — $P(\mathbf{c} \mid Q)$, $\pi$, and $\pi'$ — and check whether the structural estimate agrees with the directly observed residual.

---

## Goal

For a fixed question $Q$ ("Name a noble gas") and each focal answer $a$ in a fixed candidate universe $A$, estimate $r(a)$ by:

1. Estimating the model's conditional belief over configurations: $P(\mathbf{c} \mid C_a = 1, Q)$ and $P(\mathbf{c} \mid C_a = 0, Q)$
2. Estimating the model's generation policies: $\pi(a \mid \mathbf{c}, Q)$ and $\pi'(a \mid \mathbf{c}, Q)$
3. Computing the weighted expectations and taking their ratio

Then compare $\log \hat{r}(a)$ (structural estimate) against $s_V(a) - s_G(a) + s_{G'}(a)$ (direct observation).

---

## Step 0: Fix the Candidate Universe

$$A = \{\underbrace{\text{Helium, Neon, Argon, Krypton, Xenon, Radon, Oganesson}}_{7 \text{ noble gases}}, \underbrace{\text{Oxygen, Carbon, Sodium, Chlorine, Hydrogen, Nitrogen, Potassium}}_{7 \text{ non-noble}}\}$$

$|A| = 14$. Everything is relative to this universe — "configuration" means a binary vector over these 14 candidates. When we say "the correct answers," we mean: among these candidates, exactly this subset is correct. Not globally across all possible strings.

With $|A| = 14$ there are $2^{14} = 16{,}384$ total configurations. This is tractable for config elicitation (~4.5h per model) and requires pruning before π/π' estimation.

**Candidate selection rationale:** The 7 noble gases are the complete set. The 7 non-noble elements were chosen to maximize coverage of the negative generator's mass: for 9b-it, they capture ~78.5% of the generation probability under the neg-instructed-zero prompt (Oxygen 37%, Carbon 14%, Sodium 11%, Chlorine 6.5%, Hydrogen 4%, Nitrogen 4%, Potassium 2.4%).

**Future work:** Adding Calcium, Iron, and Copper would increase coverage to ~81% for 9b-it, but $|A| = 17$ gives $2^{17} = 131{,}072$ configs — an 8x increase in config elicitation cost (~36h). This could be revisited with importance sampling or if the pilot shows the distribution is sparse enough to prune aggressively.

With $|A| = 9$ there are $2^9 = 512$ total configurations. This is tractable.

**Key efficiency insight:** $P(\mathbf{c} \mid Q)$ does not depend on the focal answer $a$ — it is a property of the question and candidate universe. We estimate it **once**, save it, and reuse for all focal answers. Similarly, π and π' are estimated once per configuration (scoring all candidates in the pool), giving values for all focal answers simultaneously.

---

## Step 1: Focal Answers

We will compute $\hat{r}(a)$ for **all** $a \in A$. The per-answer cost is negligible since P(c|Q), π, and π' are shared. Suggested focal answers for the pilot analysis:

- **Helium** (clearly correct, high generator mass)
- **Argon** (clearly correct, moderate generator mass)
- **Hydrogen** (clearly incorrect)
- (plus all other candidates — the computation is shared)

---

## Step 2: Estimate P(c | Q) via Sequential Elicitation

This step is done **once** for the entire candidate universe and shared across all focal answers.

### Method

We do NOT ask for the probability of a full configuration in one shot. Instead, we decompose it using the chain rule. For a chosen ordering $\sigma = (x_1, x_2, \ldots, x_n)$ of ALL $n$ candidates, the full joint factorizes as:

$$P(\mathbf{c} \mid Q) = P(C_{x_1} = c_{x_1} \mid Q) \cdot \prod_{k=2}^{n} P(C_{x_k} = c_{x_k} \mid C_{x_1} = c_{x_1}, \ldots, C_{x_{k-1}} = c_{x_{k-1}}, Q)$$

The **first factor** is the marginal $P(C_{x_1} \mid Q)$, obtained from the validated **validator prompt** (`val-few-shot`).

The **remaining factors** are conditional probabilities obtained by prompting the model with a growing set of Suppose premises and scoring Yes/No.

### Prompt templates (validated)

**Step 1 — Marginal for first candidate** (val-few-shot):

```
Q: Is "Jupiter" a color of the rainbow? Answer Yes or No.
A: No

Q: Is "Mars" a planet in the solar system? Answer Yes or No.
A: Yes

Q: Is "{x_1}" a noble gas? Answer Yes or No.
A:
```

This gives $V(x_1) = P(C_{x_1} = 1 \mid Q)$ via log-odds → sigmoid.

**Steps 2 through $n$ — Sequential conditionals** (sequential Suppose):

For step $k$, the prompt is:

```
Q: Suppose "{x_1}" is [not] a noble gas. Suppose "{x_2}" is [not] a noble gas. ... Is "{x_k}" a noble gas? Answer Yes or No.
A:
```

The "[not]" is included or excluded depending on the assumed value $c_{x_j}$ for all previously fixed candidates. The phrasing uses the **sequential Suppose** format, validated through:
- Pairwise coherence testing (RMSE 0.155–0.175 for positive, 0.207–0.234 for negative factorizations)
- Triple-conditional ablation (sequential phrasing best for 2b-it, equivalent for 9b-it)

Note: we also compute $V(x_i)$ (the marginal for each candidate) from the validator prompt. These are needed in Step 6 and are cheap ($n$ forward passes total). This can be done at the start, before the chain.

### Scoring

**Step 1:** Score Yes/No token variants (5 each), take logsumexp of each group, compute log-odds. Then $V(x_1) = \text{sigmoid}(\text{logodds})$.

**Steps 2+:** Same Yes/No scoring at each step. Take $P(C_{x_k} = c_{x_k})$ as $P(\text{Yes})$ if $c_{x_k} = 1$, or $1 - P(\text{Yes})$ if $c_{x_k} = 0$.

**Full configuration probability** under ordering $\sigma$ (in log space, to avoid underflow):

$$\log P_{\text{model}}^{(\sigma)}(\mathbf{c} \mid Q) = \log P(C_{x_1} = c_{x_1} \mid Q) + \sum_{k=2}^{n} \log P(C_{x_k} = c_{x_k} \mid C_{x_1} = c_{x_1}, \ldots, Q)$$

### Cost

For one ordering, evaluating ALL $2^n = 512$ configurations:
- Step 1: 1 prompt (marginal; 2 configs branch)
- Step 2: 2 prompts (4 configs)
- Step $k$: $2^{k-1}$ prompts
- Total: $2^0 + 2^1 + \ldots + 2^{n-1} = 2^n - 1 = 511$ forward passes per ordering

Plus $n = 9$ validator prompts for the marginals (done once upfront).

With $K$ orderings: $511 \times K + 9$ forward passes total.

**For the pilot** ($K = 20$): $511 \times 20 + 9 = 10{,}229$ forward passes for configuration elicitation. This is done **once** and shared across all focal answers.

### What to save

Save to disk (e.g., `config_probs_{model}.json`):
- All $n$ marginals $V(x_i)$ (with raw log-odds)
- Per ordering $\sigma_k$: the full tree of stepwise log-probabilities (for debugging order sensitivity)
- The order-averaged $\bar{P}_K(\mathbf{c} \mid Q)$ for all $2^n$ configs
- The renormalized $\hat{P}(\mathbf{c} \mid C_a = 1, Q)$ and $\hat{P}(\mathbf{c} \mid C_a = 0, Q)$ for each $a$ (derived by filtering and renormalizing)

---

## Step 3: Order Averaging

Because the model is not perfectly coherent, configuration probabilities depend on the elicitation order $\sigma$. We average over $K$ random permutations of all $n$ candidates:

$$\bar{P}_K(\mathbf{c} \mid Q) = \frac{1}{K} \sum_{k=1}^{K} P_{\text{model}}^{(\sigma_k)}(\mathbf{c} \mid Q)$$

where each $\sigma_k$ is a random permutation of all $n$ candidates. Note that different orderings will use different candidates for the first (marginal) step, which is desirable for averaging out any first-position bias.

**How many orderings?** To be determined by convergence test. Proposed: start with $K = 20$, check whether the running average of $\hat{r}(a)$ has stabilized.

---

## Step 4: Renormalize

The order-averaged probabilities $\bar{P}_K(\mathbf{c} \mid Q)$ generally do not sum to 1 over all $2^n$ configurations. Renormalize:

$$\hat{P}(\mathbf{c} \mid Q) = \frac{\bar{P}_K(\mathbf{c} \mid Q)}{\sum_{\mathbf{c}'} \bar{P}_K(\mathbf{c}' \mid Q)}$$

Then, for any focal answer $a$, derive the conditional distributions by filtering and renormalizing:

$$\hat{P}(\mathbf{c} \mid C_a = 1, Q) = \frac{\hat{P}(\mathbf{c} \mid Q)}{\sum_{\mathbf{c}': c'_a = 1} \hat{P}(\mathbf{c}' \mid Q)} \quad \text{for } c_a = 1$$

$$\hat{P}(\mathbf{c} \mid C_a = 0, Q) = \frac{\hat{P}(\mathbf{c} \mid Q)}{\sum_{\mathbf{c}': c'_a = 0} \hat{P}(\mathbf{c}' \mid Q)} \quad \text{for } c_a = 0$$

Note: the denominator $\sum_{\mathbf{c}': c'_a = 1} \hat{P}(\mathbf{c}' \mid Q)$ is the model's implied marginal $\hat{V}(a)$. This can be compared against the directly probed $V(a)$ from the validator prompt as a **consistency check**.

---

## Step 5: Estimate π(a | c, Q) and π'(a | c, Q)

For each configuration $\mathbf{c}$, we run **two prompts**:

- **π prompt** ("Name a noble gas"): scores all candidates, pool-normalized over $S^+(\mathbf{c})$. Gives $\pi(a \mid \mathbf{c}, Q)$ for **every** $a \in S^+(\mathbf{c})$ simultaneously.
- **π' prompt** ("Name something NOT a noble gas"): scores all candidates, pool-normalized over $S^-(\mathbf{c})$. Gives $\pi'(a \mid \mathbf{c}, Q)$ for **every** $a \in S^-(\mathbf{c})$ simultaneously.

This is done **once per configuration** and the results are reused for all focal answers. Pool normalization guarantees $\sum_{a \in S^+(\mathbf{c})} \pi(a \mid \mathbf{c}, Q) = 1$ and $\sum_{a \in S^-(\mathbf{c})} \pi'(a \mid \mathbf{c}, Q) = 1$ by construction. Save the raw logprobs before normalization for debugging.

### Concrete example (focal answer = Helium)

**For D(Helium)** — config with $c_{\text{He}} = 1$, e.g., correct = {He, Ne, Ar}, incorrect = {Xe, Kr, Ra, H, O, C}:

```
Suppose "Xenon" is not a noble gas. Suppose "Krypton" is not a noble gas. Suppose "Radon" is not a noble gas. Suppose "Hydrogen" is not a noble gas. Suppose "Oxygen" is not a noble gas. Suppose "Carbon" is not a noble gas. Suppose "Helium" is a noble gas. Suppose "Neon" is a noble gas. Suppose "Argon" is a noble gas.

Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: Helium, Neon, Argon.

Q: Name a noble gas.
A:
```

Score Helium, Neon, Argon. Pool-normalize over {He, Ne, Ar}. The result for Helium is $\pi(\text{He} \mid \mathbf{c}, Q)$.

**For N(Helium)** — config with $c_{\text{He}} = 0$, e.g., correct = {Ne, Ar, Xe}, incorrect = {He, Kr, Ra, H, O, C}:

```
Suppose "Neon" is a noble gas. Suppose "Argon" is a noble gas. Suppose "Xenon" is a noble gas. Suppose "Helium" is not a noble gas. Suppose "Krypton" is not a noble gas. Suppose "Radon" is not a noble gas. Suppose "Hydrogen" is not a noble gas. Suppose "Oxygen" is not a noble gas. Suppose "Carbon" is not a noble gas.

Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: Helium, Krypton, Radon, Hydrogen, Oxygen, Carbon.

Q: Name something that is NOT a noble gas.
A:
```

Score all 6 incorrect candidates. Pool-normalize over {He, Kr, Ra, H, O, C}. The result for Helium is $\pi'(\text{He} \mid \mathbf{c}, Q)$.

### Prompt format: `suppose_pool` (validated)

For a configuration $\mathbf{c}$ with correct set $S^+(\mathbf{c})$ and incorrect set $S^-(\mathbf{c})$:

**π(a | c, Q)** — used on the $c_a = 1$ side, pool-normalized over $S^+(\mathbf{c})$ (which contains $a$):

```
Suppose "{x_1}" is not a noble gas. ... Suppose "{x_m}" is a noble gas. ...

Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: {S^+(c)}.

Q: Name a noble gas.
A:
```

**π'(a | c, Q)** — used on the $c_a = 0$ side, pool-normalized over $S^-(\mathbf{c})$ (which contains $a$):

```
Suppose "{x_1}" is a noble gas. ... Suppose "{x_m}" is not a noble gas. ...

Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: {S^-(c)}.

Q: Name something that is NOT a noble gas.
A:
```

### Premise ordering

- For π: incorrect premises first, correct premises last (recency bias aligns with the generation target)
- For π': correct premises first, incorrect premises last (same logic — the target group for π' is the incorrect set)
- Within each group, shuffled randomly per evaluation

### Order averaging for π/π'

Each evaluation of $\pi(a \mid \mathbf{c}, Q)$ or $\pi'(a \mid \mathbf{c}, Q)$ must itself be order-averaged over $K_\pi$ random premise orderings (including both group orders: correct-first and incorrect-first). This is separate from the $K$ orderings used for configuration elicitation.

$K_\pi$ to be determined by convergence test. Proposed: $K_\pi = 10$–$20$.

### Scoring

For each prompt, compute $\log P(\text{completion} \mid \text{prompt})$ for every candidate in the pool by summing per-token log-probs. Then pool-normalize using logsumexp:

$$\pi(a \mid \mathbf{c}, Q) = \frac{\exp(\text{logprob}(a))}{\sum_{b \in S^+(\mathbf{c})} \exp(\text{logprob}(b))}$$

$$\pi'(a \mid \mathbf{c}, Q) = \frac{\exp(\text{logprob}(a))}{\sum_{b \in S^-(\mathbf{c})} \exp(\text{logprob}(b))}$$

### Cost

For one configuration, one premise ordering: run 2 prompts (π and π'), scoring all $|A|$ candidates for each. That's $2 \times |A| = 18$ forward passes per config per ordering.

With $K_\pi$ orderings: $18 \times K_\pi$ per configuration.

Over all $2^n = 512$ configurations: $512 \times 18 \times K_\pi$ forward passes.

**For the pilot** ($K_\pi = 10$): $512 \times 18 \times 10 = 92{,}160$ forward passes for π/π' estimation. This is done **once** and gives π and π' for **all** focal answers simultaneously.

### What to save

Save to disk (e.g., `pi_estimates_{model}.json`):
- Per configuration $\mathbf{c}$, per premise ordering: raw logprobs for all candidates (before normalization)
- Per configuration: order-averaged, pool-normalized $\hat{\pi}(a \mid \mathbf{c}, Q)$ for all $a \in S^+(\mathbf{c})$
- Per configuration: order-averaged, pool-normalized $\hat{\pi}'(a \mid \mathbf{c}, Q)$ for all $a \in S^-(\mathbf{c})$

---

## Step 6: Compute the Expectations

For **each** $a \in A$ (all focal answers, using the shared P and π/π' estimates):

**Denominator** (expected positive policy at $a$, given $a$ is correct):

$$D(a) = \sum_{\mathbf{c}: c_a = 1} \hat{P}(\mathbf{c} \mid C_a = 1, Q) \cdot \hat{\pi}(a \mid \mathbf{c}, Q)$$

**Numerator** (expected negative policy at $a$, given $a$ is incorrect):

$$N(a) = \sum_{\mathbf{c}: c_a = 0} \hat{P}(\mathbf{c} \mid C_a = 0, Q) \cdot \hat{\pi}'(a \mid \mathbf{c}, Q)$$

where $\hat{P}(\mathbf{c} \mid C_a = 1, Q)$ is derived from $\hat{P}(\mathbf{c} \mid Q)$ by filtering to $c_a = 1$ configs and renormalizing (Step 4).

**Structural estimate:**

$$\hat{r}(a) = \frac{N(a)}{D(a)}, \qquad \log \hat{r}(a) = \log N(a) - \log D(a)$$

**Save** D(a), N(a), and $\hat{r}(a)$ for all $a \in A$.

---

## Step 7: Compare Against Direct Observation

The directly observed residual is:

$$\log r_{\text{obs}}(a) = s_V(a) - s_G(a) + s_{G'}(a)$$

where $s_V$, $s_G$, $s_{G'}$ come from the validated prompt templates (val-few-shot, instructed-zero, neg-instructed-zero).

The key test: **does $\log \hat{r}(a) \approx \log r_{\text{obs}}(a)$ across focal answers?**

### Connection to observed scores via V(a)

The structural decomposition connects to the observed scores through $V(a)$:

$$G(a) = V(a) \cdot D(a), \qquad G'(a) = (1 - V(a)) \cdot N(a)$$

So $\log r = \log N - \log D = \log G'(a) - \log(1 - V(a)) - \log G(a) + \log V(a) = s_V - (s_G - s_{G'})$, recovering Eq. 6. This means $V(a)$ is the bridge between the per-side conditional expectations and the full marginal quantities.

### Secondary analyses

- Is $\hat{r}(a)$ roughly constant across answers? Across answer classes (correct vs incorrect)?
- Which configurations $\mathbf{c}$ carry most of the weight in $\hat{P}$? Is the distribution sparse?
- Does the structural decomposition reconstruct the raw generator probability: $\hat{G}(a) = V(a) \cdot D(a) \approx G_{\text{obs}}(a)$? And similarly $\hat{G}'(a) = (1 - V(a)) \cdot N(a) \approx G'_{\text{obs}}(a)$?

---

## Step 8: Diagnostics

### A. Order sensitivity

For a handful of configurations, check how much $P_{\text{model}}^{(\sigma)}(\mathbf{c})$ varies across orderings. If wildly unstable, the elicitation is fragile and $K$ must increase.

### B. Mass concentration

After renormalization, inspect which configurations get most weight. If nearly all mass concentrates on a few configs, the effective summation is sparse, which is both computationally helpful and scientifically informative.

### C. Marginal consistency check

The implied marginal from the config elicitation is $\hat{V}(a) = \sum_{\mathbf{c}: c_a = 1} \hat{P}(\mathbf{c} \mid Q)$. Compare this against the directly probed $V(a)$ from the validator prompt for each candidate $a$. If these diverge significantly, the sequential elicitation is not preserving the model's marginal beliefs. Report $|\hat{V}(a) - V(a)|$ for all $a \in A$.

### D. Positive generator reconstruction

The structural decomposition predicts the observed positive generation probabilities. Compare for each $a \in A$:

$$\hat{G}(a) = \sum_{\mathbf{c}:\, c_a = 1} \hat{P}(\mathbf{c} \mid Q) \cdot \hat{\pi}(a \mid \mathbf{c}, Q) \quad \stackrel{?}{\approx} \quad G_{\text{obs}}(a)$$

where $G_{\text{obs}}(a) = \exp(s_G(a))$ from the plain `instructed-zero` prompt (no configuration conditioning).

This checks whether the decomposition $G(a) = V(a) \cdot D(a)$ is internally consistent with the model's actual unconditioned generation behavior. If $\hat{G}$ and $G_{\text{obs}}$ are completely disconnected, distrust the decomposition.

### E. Negative generator reconstruction

The analogous check for the negative generator. Compare for each $a \in A$:

$$\hat{G}'(a) = \sum_{\mathbf{c}:\, c_a = 0} \hat{P}(\mathbf{c} \mid Q) \cdot \hat{\pi}'(a \mid \mathbf{c}, Q) \quad \stackrel{?}{\approx} \quad G'_{\text{obs}}(a)$$

where $G'_{\text{obs}}(a) = \exp(s_{G'}(a))$ from the `neg-instructed-zero` prompt (no configuration conditioning).

This checks whether the decomposition $G'(a) = (1 - V(a)) \cdot N(a)$ holds. Both D and E must pass for the structural estimate of $r(a,Q)$ to be trustworthy — a failure in either side means the corresponding conditional expectation ($D$ or $N$) is not capturing what the unconditioned generator actually does.

### F. Convergence in K

Plot $\hat{r}(a)$ as a function of $K$ (number of orderings). If it hasn't stabilized by $K = 20$, increase $K$.

---

## Computational Budget

### Pilot specification

| Parameter | Value |
|---|---|
| Question | "Name a noble gas" |
| Candidate universe $A$ | 14 elements (7 noble + 7 non-noble) |
| Focal answers | all 14 (shared computation) |
| Config elicitation orderings $K$ | 20 |
| π/π' premise orderings $K_\pi$ | 10 |

### Forward pass count

| Component | Formula | Count |
|---|---|---|
| Marginals (validator) | $n$ (one per candidate) | 14 |
| Config elicitation | $(2^{14} - 1) \times K$ | 327,660 |
| π/π' estimation (after pruning) | $N_{\text{surviving}} \times 2 \times |A| \times K_\pi$ | ~28,000 (est. ~100 configs) |
| **Total** | | **~356K** |

All computation is shared across focal answers — no per-answer cost. Config elicitation is the bottleneck at ~4.5 hours per model (at ~50ms per forward pass). π/π' estimation is fast after pruning (~23 min).

### Pruning low-probability configurations (required)

With $2^{14} = 16{,}384$ configs, running π/π' on all of them is infeasible ($16{,}384 \times 2 \times 14 \times 10 \approx 4.6$M passes). After config elicitation, prune configurations with $\hat{P}(\mathbf{c}) < \epsilon$ (e.g., $\epsilon = 10^{-4}$). We expect the distribution to be highly sparse — if ~100 configs survive, the π/π' cost drops to ~28K passes.

---

## Validated Prompt Templates (Summary)

All templates below have been validated through systematic ablation studies. See `docs/quality_constraint_testing.md` for full experimental history.

### Generator $s_G$: `instructed-zero`

```
Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else.

Q: Name a noble gas.
A:
```

Score type: log-probability (sum of per-token log-probs).

### Negative generator $s_{G'}$: `neg-instructed-zero`

```
Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else.

Q: Name something that is NOT a noble gas.
A:
```

Score type: log-probability.

### Validator $s_V$: `val-few-shot`

```
Q: Is "Jupiter" a color of the rainbow? Answer Yes or No.
A: No

Q: Is "Mars" a planet in the solar system? Answer Yes or No.
A: Yes

Q: Is "{X}" a noble gas? Answer Yes or No.
A:
```

Score type: log-odds (log P(Yes) − log P(No), each aggregated over 5 token variants via logsumexp).

### Configuration elicitation: val-few-shot marginal + sequential Suppose chain

**Step 1 (marginal for first candidate $x_1$ in ordering):** val-few-shot prompt (same as validator above, substituting $x_1$).

**Steps 2+ (conditionals):**

```
Q: Suppose "{x_1}" is [not] a noble gas. Suppose "{x_2}" is [not] a noble gas. ... Is "{x_k}" a noble gas? Answer Yes or No.
A:
```

Score type: log-odds → sigmoid → stepwise probability. Multiply all steps (in log space) to get $P(\mathbf{c} \mid Q)$.

### π(a|c,Q): `suppose_pool`

```
Suppose "{x_1}" is not a noble gas. ... Suppose "{x_m}" is a noble gas. ...

Answer the following question with a single word. Write only the full name of the element (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: {correct_set}.

Q: Name a noble gas.
A:
```

Premise ordering: incorrect first, correct last (default); average over both group orders.
Score type: log-probability → pool-normalize over correct set.

### π'(a|c,Q): `suppose_pool`

```
Suppose "{x_1}" is a noble gas. ... Suppose "{x_m}" is not a noble gas. ...

Answer the following question with a single word. Write only the full name of the element or substance (e.g. "Sodium", "Chlorine"). Do not write anything else. Your answer MUST be one of: {incorrect_set}.

Q: Name something that is NOT a noble gas.
A:
```

Premise ordering: correct first, incorrect last (default); average over both group orders.
Score type: log-probability → pool-normalize over incorrect set.

---

## Execution Plan

### Phase 1: Convergence test (pre-pilot)

Before the full pilot, test how many orderings are needed for stability.

1. Run config elicitation with $K = 30$ orderings (random permutations of all 9 candidates)
2. Run π/π' with $K_\pi = 20$ orderings on the top-weighted configs
3. Plot running average of $\hat{r}(\text{Helium})$ and $\hat{r}(\text{Hydrogen})$ vs $K$ and vs $K_\pi$
4. Determine the minimal $K$ and $K_\pi$ for which the estimates stabilize (e.g., relative change < 5%)

### Phase 2: Pilot ($|A| = 9$, all focal answers)

1. Run config elicitation (once, shared)
2. Inspect mass concentration; prune low-weight configs
3. Run π/π' estimation on surviving configs (once, shared)
4. Compute $\hat{r}(a)$ for all $a \in A$
5. Compare $\log \hat{r}(a)$ against $\log r_{\text{obs}}(a) = s_V - (s_G - s_{G'})$
6. Run all diagnostics (order sensitivity, mass concentration, marginal consistency, generator reconstruction, convergence)
7. Analyze: is $\hat{r}$ roughly constant? Constant within classes? Do correct vs incorrect answers differ?

### Phase 3: Scale (if pilot succeeds)

- Expand to a second question domain (e.g., AmbigQA)
- Test on both gemma-2-2b-it and gemma-2-9b-it
- Explore larger candidate universes (may require importance sampling)

---

## What This Procedure Is

This is not "the true hidden belief state of the LLM." It is an elicited, order-averaged, renormalized approximation to the model's conditional distribution over exact configurations, together with directly scored within-pool choice probabilities. That sounds fussy, but it is the right level of honesty.

---

## Open Questions

1. **Is $K = 20$ sufficient for config elicitation?** The convergence test will answer this.
2. **How sparse is $\hat{P}(\mathbf{c})$?** If very sparse, the pruning optimization dramatically reduces cost. If not, we may need to consider importance sampling or other approximations for larger candidate sets.
3. **Does the chain-rule factorization introduce compounding errors?** With 8 multiplicative steps, small per-step errors could compound. The renormalization partially addresses this, but the convergence test should check.
4. **Scaling beyond $|A| = 9$.** With $|A| = 15$, we'd have $2^{14} = 16{,}384$ configs per side — likely too many without pruning or sampling. Importance sampling over configs (biased toward high-P configs discovered during elicitation) may be needed.
