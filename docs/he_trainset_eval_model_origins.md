# HumanEval train-set eval — WHERE EACH MODEL LIVES (read before touching)

The gemma-4-31B-it humaneval checkpoints are split across **two directories** because of
a consolidation move. This note exists so nobody (including future-me) wastes time
"looking in the wrong place" again — which already happened once on 2026-06-17.

## The split (gemma-4-31B-it, humaneval)

| Settings | Directory (on mll, /datastor2) | How they got there |
|---|---|---|
| **s2** (RankAlign), **s4** (FLORA-PMI / New+fsx+tc) | **`models2-rerun-wandb/`** | Trained in the *earlier* wandb-rerun run, then **MOVED** here (user-approved, 2026-06-17) when consolidating the finetuned gemma adapters out of `gemma-4-models-mll-tmp/`. |
| **s1** (SFT), **s3** (New+fsx), **s7** (FLORA-Neg), **s13** (SFT+cft) | **`gemma-4-models-mll-tmp/`** (upper) and **`gemma-4-models-mll-tmp-multi/`** (multi) | The *current* run (launched 2026-06-16); the genctx launcher's default model dirs. |

**Both s2 and s4 have epoch0 / epoch1 / epoch2 present for BOTH upper and multi** (verified
2026-06-17). Exact dir names (delta tag varies, so globs use `delta*`):

- s2 upper/multi: `v7-google--gemma-4-31B-it-delta2.14-epoch{0,1,2}--humaneval-v2.1correct-{upper,multi}-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1`
- s4 upper: `...-delta3.26-epoch{0,1,2}--...-correct-upper-...--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1`
- s4 multi: `...-delta3.71-epoch{0,1,2}--...-correct-multi-...--tc-self--...` (same suffix as s4 upper)

## qwen-3.5-9B (for when train-eval extends to qwen)

All qwen humaneval checkpoints (s1/s2/s3/s4/s7/s13, **merged** full models, `..._merged`) live
in **`models2-rerun-wandb/`**.

## What the train-set eval launcher does with this

`scripts/mll_he_trainset_eval.sbatch` (settings s2/s4) globs the s2/s4 checkpoints from
**`models2-rerun-wandb/`**. A future s1/s3/s7/s13 train-eval must instead glob
`gemma-4-models-mll-tmp[-multi]/`. The launcher header repeats this so it can't be missed.

## Outputs

Train-set scores (`--train` split) are written to a **dedicated** dir
`/datastor2/jdr/rankalign/outputs-he-trainset/` (kept separate from the `_test_` scores in
`outputs_gemma4_mll_tmp[-multi]/`). Filenames carry `_train_`, the epoch (`epoch0/1/2`) or the
base model name, and the dataset (`correct-upper`/`correct-multi`), so all 16 cells are distinguishable.
