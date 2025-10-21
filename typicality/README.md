# Typicality Analysis

This directory contains scripts for analyzing the role of typicality in the generator-validator gap.

## Theory

The hypothesis is that generator probabilities should integrate both:
1. **Validator correctness** (semantic truth)
2. **Typicality** (how expected/typical the completion is)

Formally: `generator_prob ∝ validator_score × typicality_score`

Or in log-space: `log P_gen(y|x) ≈ log P_val(correct) + log P_typicality(y) + const`

## Typicality Measures

We use GPT-2 as an independent small LM to measure typicality:

1. **P_gpt2(noun2)** - Unconditional probability (lexical typicality)
2. **P_gpt2(noun1)** - Unconditional probability of the hyponym
3. **P_gpt2(noun2 | "noun1 is a kind of")** - Conditional probability (contextual typicality)

## Workflow

### Step 1: Generate eval.py debug output

First, run eval.py with the debug flag to save generator and discriminator scores:

```bash
cd ../scripts
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model google/gemma-2-2b \
    --task hypernym \
    --seed 0 \
    --split_type random \
    --debug_save_values
```

This creates: `../outputs/debug_values_hypernym_logodds.csv`

### Step 2: Compute GPT-2 typicality scores

```bash
cd ../typicality
python compute_gpt2_typicality.py \
    --seed 0 \
    --split_type random \
    --output gpt2_typicality_scores.csv
```

This creates: `gpt2_typicality_scores.csv` with typicality measures.

### Step 3: Merge the data

```bash
python merge_data.py \
    --typicality gpt2_typicality_scores.csv \
    --eval_output ../outputs/debug_values_hypernym_logodds.csv \
    --output merged_analysis_data.csv
```

This creates: `merged_analysis_data.csv` with everything combined.

**Important**: The merge script verifies that:
- Row counts match
- Noun pairs (noun1, noun2) match exactly between files
- Ground truth labels match
- Taxonomic labels match

If verification fails, the script will raise an error showing which rows don't match.

### Step 4: Analyze relationships

```bash
python analyze_typicality.py \
    --data merged_analysis_data.csv \
    --output_dir analysis_results/
```

(To be implemented: statistical analysis and visualizations)

## Quick Run

Use the convenience script to run all steps:

```bash
./run_typicality_analysis.sh google/gemma-2-2b
```

## Files

- `compute_gpt2_typicality.py` - Compute GPT-2 typicality scores
- `merge_data.py` - Merge typicality scores with eval output
- `analyze_typicality.py` - Statistical analysis (TODO)
- `run_typicality_analysis.sh` - Convenience script to run full pipeline
- `README.md` - This file

## Output Format

The merged CSV contains:
- `index` - Example index
- `noun1` - Hyponym (e.g., "corgi")
- `noun2` - Hypernym (e.g., "dog")
- `taxonomic` - Ground truth label ("yes"/"no")
- `ground_truth` - Binary label (1/0)
- `gen_score` - Generator log-odds from target LLM
- `disc_score` - Discriminator log-odds from target LLM
- `log_prob_noun2` - P_gpt2(noun2) unconditional
- `log_prob_noun1` - P_gpt2(noun1) unconditional
- `log_prob_noun2_given_context` - P_gpt2(noun2 | context) conditional

