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

## EXACT checkpoint directory names — the NEW runs (enumerated 2026-06-19)

Verified by listing the live dirs on mll. Every scenario has **epoch0 / epoch1 / epoch2**.
**Settings are disambiguated by recipe tokens, NOT by the delta tag** — the delta is *not*
a unique key (e.g. qwen-multi s1/s3/s4/s7 all share `delta0.92`). To select a setting, glob on
its token signature below; the `{0,1,2}` is the epoch.

Token signatures (the reliable discriminator):

| Setting | Token signature in the dir name |
|---|---|
| s1  (SFT)          | `pref0.0` + `vallogodds` + `labelonly0.1`, **no** `cft`, **no** `force-same-x` |
| s2  (RankAlign)    | `full-completion--semi0.1--fix1` only (no pref/cft/fsx/tc) |
| s3  (New+fsx)      | `force-same-x` + `vallogodds` + `semi0.1`, **no** `tc-*` |
| s4  (self-tc / FLORA-PMI) | `tc-self` + `force-same-x` + `semi0.1` |
| s7  (neg-tc / FLORA-Neg)  | `tc-neg`  + `force-same-x` + `semi0.1` |
| s13 (SFT+cft)      | `pref0.0` + `cft` + `labelonly0.1` |

### gemma-4-31B-it, NEW settings (LoRA adapter dirs)

`gemma-4-models-mll-tmp/` (upper) and `gemma-4-models-mll-tmp-multi/` (multi). Common stem:
`v7-google--gemma-4-31B-it-delta{D}-epoch{0,1,2}--humaneval-v2.1correct-{upper|multi}-all--d2g--random--alpha1.0--full-completion--<setting tokens>--fix1`

| Setting | upper delta | multi delta | setting-token tail |
|---|---|---|---|
| s1  | `delta3.02` | `delta3.24` | `--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1` |
| s3  | `delta3.26` | `delta3.71` | `--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1` |
| s7  | `delta3.26` | `delta3.71` | `--tc-neg--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1` |
| s13 | `delta2.14` | `delta2.15` | `--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1` |

(NB: s3 and s7 share the same delta within a dataset — `tc-neg` is the only difference.)

### qwen-3.5-9B, ALL six settings (merged full models, suffix `_merged`)

`models2-rerun-wandb/`. Common stem:
`v7-Qwen--Qwen3.5-9B-delta{D}-epoch{0,1,2}--humaneval-v2.1correct-{upper|multi}-all--d2g--random--alpha1.0--full-completion--<setting tokens>--fix1_merged`

| Setting | upper delta | multi delta | setting-token tail |
|---|---|---|---|
| s1  | `delta0.97` | `delta0.92` | `--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1` |
| s2  | `delta0.72` | `delta0.72` | *(none — bare `--semi0.1`)* |
| s3  | `delta0.94` | `delta0.92` | `--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1` |
| s4  | `delta0.94` | `delta0.92` | `--tc-self--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1` |
| s7  | `delta0.94` | `delta0.92` | `--tc-neg--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1` |
| s13 | `delta0.76` | `delta0.77` | `--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1` |

## What the train-set eval launcher does with this

`scripts/mll_he_trainset_eval.sbatch` (settings s2/s4) globs the s2/s4 checkpoints from
**`models2-rerun-wandb/`**. A future s1/s3/s7/s13 train-eval must instead glob
`gemma-4-models-mll-tmp[-multi]/`. The launcher header repeats this so it can't be missed.

## Outputs

Train-set scores (`--train` split) are written to a **dedicated** dir
`/datastor2/jdr/rankalign/outputs-he-trainset/` (kept separate from the `_test_` scores in
`outputs_gemma4_mll_tmp[-multi]/`). Filenames carry `_train_`, the epoch (`epoch0/1/2`) or the
base model name, and the dataset (`correct-upper`/`correct-multi`), so all 16 cells are distinguishable.
