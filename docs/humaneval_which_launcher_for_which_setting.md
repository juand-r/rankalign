# Which launcher trained which humaneval setting (gemma-4 & qwen-3.5)

For future-you who forgot. The humaneval v2.1correct format glues the function body
directly onto the signature with no separator, and on a fraction of items the tokenizer
merges that seam so the completion tokenizes differently in-context vs standalone. The
trainer's `_check_tail` guard **hard-aborts** on such an item.

- **`scripts/ranking_loss_ref_fix.py`** — the ORIGINAL trainer. Aborts on a seam item.
- **`scripts/ranking_loss_ref_fix_genctx.py`** — an isolated COPY that adds a pair-level
  drop filter (skips the seam items). The original is left untouched. When no item
  mismatches, the filter is a no-op and the result is identical to fix.py.
  See `docs/qwen_humaneval_genctx_fix.md`.

The mismatch rate differs by tokenizer: **qwen ~6%** (training aborts without the filter),
**gemma ~0.3%** (the few seam items happened not to be sampled, so gemma trained fine on
fix.py). That asymmetry is why the launchers split the way they do.

## The mapping (this is the part you'll forget)

| Model | Settings | Launcher | Trainer it calls |
|---|---|---|---|
| **gemma-4-31B-it** | **s2, s4** | `pod-setup-train-scripts-gemma-4/mll_g4_train_eval.sbatch` | `ranking_loss_ref_fix.py` (original) |
| **gemma-4-31B-it** | **s1, s3, s7, s13** | `pod-setup-train-scripts-gemma-4/mll_g4_train_eval_genctx.sbatch` | `ranking_loss_ref_fix_genctx.py` (drop filter) |
| **qwen-3.5-9B** | **all six** (s1,s2,s3,s4,s7,s13) | `scripts/mll_qwen35_he_train_eval.sbatch` | `ranking_loss_ref_fix_genctx.py` (drop filter) |

Both humaneval datasets (`humaneval-v2.1correct-upper` and `-multi`) use the **same**
launchers — pick the dataset with the `HE_TASK` env var.

## Why it's split this way (reproducibility)

Each launcher reproduces *exactly* what produced its results:

- **gemma s2/s4** were trained with the original `fix.py` and completed. So their launcher
  (`mll_g4_train_eval.sbatch`) is kept **byte-for-byte as it ran** — original trainer,
  s2/s4 only, `REPO=/datastor1`. Do **not** run s1/s3/s7/s13 there.
- **gemma s1/s3/s7/s13** are new. `s3/s7` use `--force-same-x`, which pairs items
  differently and could surface a seam item mid-run and abort. So they get the genctx
  launcher (`..._genctx.sbatch`), which also fixes `REPO=/datastor2` (the original
  `/datastor1` is nearly full and caused an ENOSPC that killed 12 evals on 2026-06-16).
- **qwen** could not train s2/s4 on `fix.py` at all (it aborted on the 6% mismatch). The
  qwen s2/s4 results that exist were produced with `genctx`. So the single qwen launcher
  calls `genctx` for **all** settings — that is its faithful record; reverting it to
  `fix.py` would make it non-reproducible.

## One-line rule

- gemma **s2/s4** → `mll_g4_train_eval.sbatch` (fix.py).
- gemma **s1/s3/s7/s13** → `mll_g4_train_eval_genctx.sbatch` (genctx).
- qwen **everything** → `mll_qwen35_he_train_eval.sbatch` (genctx).
