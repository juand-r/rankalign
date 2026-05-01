# Bayesian Coherence of LLM Validation and Generation

## Motivation

One way to use an LLM as a **validator** (discriminator) is to prompt it with "Is A a valid answer to Q?" and read off the probability of "Yes." Another way is to use the LLM as a **generator** — prompt it with "Give a valid answer to Q" — and look at how much probability mass it places on A.

A natural question: are these two modes of querying consistent with each other? Can we relate them via Bayes' rule? And if so, does the LLM actually satisfy this consistency in practice?

Below we build a formal framework for answering these questions.

---

## Setup

### Sample Space

Let Ω = **Q** × **A** × {0, 1}, where:

- **Q** is the set of all possible questions,
- **A** is the set of all possible answers,
- V ∈ {0, 1} indicates whether the answer is valid (1) or invalid (0).

A point (q, a, v) ∈ Ω represents: "the question is q, the answer is a, and it is/isn't valid."

### Probability Measure

P is a joint distribution over Ω representing the **LLM's beliefs**. This is a subjective probability — in reality, validity is deterministic given (q, a), but the LLM doesn't know this with certainty. P captures its uncertainty.

### Key Conditionals

All derived from the single joint P:

| Expression | Interpretation | How to access via LLM |
|---|---|---|
| P(V=1 \| A=a, Q=q) | LLM's belief that a is a valid answer to q | Prompt: "Is a a valid answer to q? Yes or no." |
| P(A=a \| Q=q, V=1) | Distribution over valid answers | Prompt: "Give a valid answer to q." |
| P(A=a \| Q=q, V=0) | Distribution over invalid answers | Prompt: "Give an invalid answer to q." |
| P(V=1 \| Q=q) | Prior belief that a random answer to q is valid | Not directly prompted — a marginal of the joint |
| P(A=a \| Q=q) | Marginal over answers (valid and invalid) | Not directly prompted — a marginal of the joint |

### Bayes' Rule

Since everything derives from one joint P, Bayes' rule holds as an identity:

$$P(V=1 \mid A=a, Q=q) = \frac{P(A=a \mid Q=q, V=1) \; P(V=1 \mid Q=q)}{P(A=a \mid Q=q)}$$

---

## The Marginals

The marginal distributions are defined by summing out variables from the joint:

$$P(A=a \mid Q=q) = \sum_{v} P(A=a, V=v \mid Q=q)$$

$$P(V=1 \mid Q=q) = \sum_{a} P(A=a, V=1 \mid Q=q)$$

These can equivalently be written as a mixture:

$$P(A=a \mid Q=q) = P(V\!=\!1 \mid Q) \cdot P(A=a \mid Q, V\!=\!1) + P(V\!=\!0 \mid Q) \cdot P(A=a \mid Q, V\!=\!0)$$

Note that the mixture form is a **consequence** of the joint, not a definition. Treating it as a definition introduces circularity.

Additionally, P(V=1 | Q) + P(V=0 | Q) = 1. This is a consistency requirement: the total probability of validity and invalidity must exhaust all possibilities.

---

## Constructing the Joint

A key problem: we cannot easily extract P(A=a | Q=q) from the LLM by prompting. Prompting with just a question (e.g., "Q: What is the capital of Australia? A:") effectively asks for a *valid* answer, because the LLM has been trained to be helpful. This gives us something close to P(A | Q, V=1), not the true marginal.

The solution: **don't try to extract the marginal.** Instead, construct it.

We can directly extract two distributions from the LLM:

- **gen(a)** = P(A=a | Q=q, V=1) — prompt for a valid answer
- **gen'(a)** = P(A=a | Q=q, V=0) — prompt for an invalid answer

Then we choose a free parameter:

- **λ(Q)** = P(V=1 | Q=q) — the prior probability that a random answer is valid, which may depend on the question

And **define** the marginal:

$$P(A=a \mid Q=q) = \lambda \cdot \text{gen}(a) + (1 - \lambda) \cdot \text{gen}'(a)$$

This is consistent by construction. No coherence test is needed for the marginal, because we built the joint from its components.

---

## Example: "What is the capital of Australia?"

The LLM's joint assigns mass to various (answer, validity) pairs:

| Answer | V | Mass | Why |
|---|---|---|---|
| Canberra | 1 | high | Correct answer, appears in reliable sources |
| Sydney | 0 | decent | Wrong, but a very common misconception |
| Melbourne | 0 | some | Historically was the capital |
| Perth | 0 | tiny | Rarely associated with this question |
| Pizza | 0 | ≈ 0 | Never comes up in this context |

For this question, λ = P(V=1 | Q) might be relatively low — many people get it wrong, so the LLM's training data contains a lot of incorrect answers.

**Sydney** has high gen'(a) (likely under "give an invalid answer") but near-zero gen(a) (unlikely under "give a valid answer"). So the likelihood ratio gen/gen' is tiny, and the discriminator correctly rejects it.

**Canberra** has high gen(a) and low gen'(a). The likelihood ratio is large, and the discriminator scores it highly.

Contrast with Q = "What is the capital of France?" — almost nobody gets this wrong, so λ is close to 1 and the discriminator is lenient.

---

## The Empirical Test

### Shorthand

For a fixed question q, define for each candidate answer a:

- **val(a)** = P(V=1 | A=a, Q=q) — the validator score
- **gen(a)** = P(A=a | Q=q, V=1) — the generator score
- **gen'(a)** = P(A=a | Q=q, V=0) — the "wrong" generator score
- **Z(a)** = λ · gen(a) + (1−λ) · gen'(a) — the normalizer

### Bayes' Rule Constraint

The framework requires:

$$\text{val}(a) = \frac{\lambda \cdot \text{gen}(a)}{Z(a)}$$

### Odds Form

Rearranging, we get Bayes' rule in odds form:

$$\frac{\text{val}(a)}{1 - \text{val}(a)} = \frac{\lambda}{1 - \lambda} \cdot \frac{\text{gen}(a)}{\text{gen}'(a)}$$

The odds of validity equal the **prior odds** times the **likelihood ratio**.

### The Test

For a fixed question Q:

1. Choose a collection of candidate answers {a₁, a₂, ..., aₙ}.
2. For each aᵢ, extract val(aᵢ), gen(aᵢ), and gen'(aᵢ) from the LLM.
3. Plot val(aᵢ) / (1 − val(aᵢ)) against gen(aᵢ) / gen'(aᵢ).

**If the LLM is Bayesian-coherent**, the points should lie on a straight line through the origin with slope λ / (1 − λ).

The slope directly recovers the LLM's implicit prior on answer validity for that question.

---

## Summary

The framework reduces to three claims:

1. The LLM has an implicit joint P over (question, answer, validity).
2. Different prompts extract different conditionals of this joint.
3. Bayes' rule is a consistency condition these conditionals must satisfy.

The only free parameter is λ(Q) — the prior on validity — which controls how aggressive the discriminator is. The empirical test (linearity of the odds plot) provides a direct way to check whether any given LLM satisfies this coherence condition.
