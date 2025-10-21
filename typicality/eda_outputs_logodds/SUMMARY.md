# EDA Summary: Typicality and Generator-Validator Gap

**Data**: `merged_data_google-gemma-2-2b_random_seed0-logodds.csv`  
**Date**: 2025-10-19

## Key Findings

### 1. Typicality Scores Correlate with Generator Scores

- **P(noun2|context)** has the strongest correlation with `gen_score` (r=0.341, ρ=0.367)
- **P(noun2)** has moderate correlation with `gen_score` (r=0.175, ρ=0.156)
- **P(noun1)** has weak correlation with `gen_score` (r=0.076, ρ=0.080)

**Interpretation**: The conditional probability of the hypernym given the context is most predictive of what the generator produces.

### 2. G-V Gap Differs Dramatically by Label

- **Negative examples**: G-V gap = -10.04 ± 3.68
- **Positive examples**: G-V gap = -4.75 ± 2.47

The generator struggles much more with negative examples (non-hypernyms) compared to the validator.

### 3. G-V Gap Correlates with Typicality

- **P(noun2|context)**: r = 0.346 (strongest)
- **P(noun2)**: r = 0.183
- **P(noun1)**: r = 0.087

**Interpretation**: More typical hypernyms (in context) reduce the G-V gap.

### 4. Theoretical Model: Gen ≈ Disc + Typicality

Testing the additive model in log space:
```
gen_score ≈ α + β₁*disc_score + β₂*typicality
```

**Best model** (using P(noun2|context)):
```
gen_score = -1.15 + 5.63*disc_score + 0.24*P(noun2|context)
```
- **R² = 0.615** (compared to 0.583 for Disc alone)
- **ΔR² = 0.032** (3.2% additional variance explained)

### 5. Residual Analysis

Adding typicality to the model:
- Explains **7.7% of residual variance** after accounting for discriminator
- **Larger effect for positive examples**: ΔR² = 0.082 vs 0.060 for negatives
- **Negative examples harder to model**: Baseline R² = 0.203 vs 0.301 for positives

### 6. Split Analysis

**Negative Examples** (n=513):
- Gen-Disc correlation: 0.451
- P(noun2|context) → Gen: r = 0.287
- P(noun2|context) → Disc: r = 0.096

**Positive Examples** (n=505):
- Gen-Disc correlation: 0.548
- P(noun2|context) → Gen: r = 0.404 (stronger!)
- P(noun2|context) → Disc: r = 0.227

**Interpretation**: For positive examples, typicality has a stronger relationship with both generator and validator, but especially the generator.

## Implications

1. **Typicality matters**: Even a small LM (GPT-2) captures typicality information that helps explain generator behavior beyond what the validator predicts.

2. **Conditional typicality is key**: P(noun2|context) outperforms unconditional probabilities, suggesting context-sensitive typicality is what matters.

3. **G-V gap is partly typicality-driven**: The gap correlates with typicality, suggesting generators may be biased toward typical completions even when incorrect.

4. **Room for improvement**: The model only explains ~61% of variance, suggesting other factors (semantic relatedness, world knowledge, etc.) also play a role.

## Generated Plots

1. `correlation_matrices.png` - Pearson and Spearman correlations
2. `gen_vs_disc.png` - Generator vs Validator scores
3. `typicality_vs_scores.png` - Typicality vs Gen/Disc (6 subplots)
4. `gv_gap_vs_typicality.png` - G-V Gap vs typicality scores
5. `theoretical_model_fit.png` - Predicted vs Actual for additive models
6. `residual_analysis.png` - Residual analysis and Q-Q plot

## Next Steps

1. **Repeat with logprobs**: Same analysis with `--use_full_completion_logprobs`
2. **Test on other tasks**: Does typicality generalize to trivia-qa, swords, etc.?
3. **Try other small LMs**: Compare GPT-2 to Pythia, OPT, etc.
4. **Investigate outliers**: Which examples have large residuals?
5. **Causal analysis**: Does RankAlign reduce typicality bias?

