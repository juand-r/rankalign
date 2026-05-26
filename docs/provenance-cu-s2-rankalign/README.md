# RankAlign (Method 2) training provenance — pod `cu-s2-rankalign`

Captured 2026-05-26 from the TAUR RunPod pod **`cu-s2-rankalign`** (id `ptxhxvshbbe1ae`,
1× H100 SXM) before the pod was retired. This directory preserves the training logs,
config, and pod-side upload helper for the **RankAlign (Method 2)** gemma-4-31B-it run on
**humaneval-v2.1correct-upper**, so the run can be understood and reproduced later.

> Everything here is logs/config/scripts — **the trained models themselves are on HuggingFace**
> (see below). Nothing model-weight-sized is stored in this repo.

---

## What the run produced

Three LoRA adapters (one per epoch), each `adapter_model.safetensors` = **489,840,816 bytes**,
base model **`google/gemma-4-31B-it`**. All three are on HuggingFace (`latkes`), byte-exact
verified against the pod on 2026-05-26:

| Epoch | HF repo (latkes) | Notes |
|-------|------------------|-------|
| 0 | `rankalign-v7-g4-31b-d214-e0-cu-all-sm0.1-fix1` | auto-uploaded during the run |
| **1** | `rankalign-v7-g4-31b-d214-e1-cu-all-sm0.1-fix1` | **the evaluated checkpoint** (see "Evaluation") |
| 2 | `rankalign-v7-g4-31b-d214-e2-cu-all-sm0.1-fix1` | manually backed up 2026-05-26 (see note) |

On the pod the adapters were named (long form):
`v7-google--gemma-4-31B-it-delta2.14-epoch{N}--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1`

**Why e0/e1 were on HF but e2 was not:** `upload_cu_adapters.py` ran on the pod and pushed
e0 + e1 (`logs/hf_upload_ep01.log` → "Uploaded 2/2 repos"). It did **not** push e2 because the
training wrapper raised a **false** `FATAL: epoch2 adapter missing` post-check
(`logs/run_s2.log`) — even though the epoch2 adapter was in fact written correctly. e2 was
backed up by hand on 2026-05-26 and verified byte-identical to e0/e1.

---

## How it was trained (exact)

The RankAlign objective = **preference-loss-only** (no NLL terms), semi-supervised 0.1, no
typicality-correction, no force-same-x. Verbatim training flags from `logs/run_s2.log`
(SETTING 2 = "RankAlign"):

```
--model google/gemma-4-31B-it --num_epochs 3 --task humaneval-v2.1correct-upper \
--train_g_or_d g --split_type random \
--nll_validator_weight 0 --nll_generator_weight 0 --preference_loss_weight 1 \
--all --delta 0.15 --semi-supervised 0.1 --disc-shots zero \
--lora --gradient_checkpointing --models-dir /workspace/models_g4it \
--total_samples 5110 --no-upload-hf
```

- Training entrypoint: the repo's RankAlign training script (`scripts/ranking_loss_ref_fix.py`
  per repo convention) invoked by the on-pod wrapper `run_s2.sh`. The wrapper itself was not
  preserved (it lived on the pod), but the flags above are the complete, verbatim invocation.
- **`--delta 0.15` is the *argument*, not the realized delta.** With per-prompt shape-budget
  mode, the realized `delta_used ≈ 2.137` was computed from the validator-logprob score spread
  (p5–p95 = 21.374, /10 bins) — hence **`delta2.14`** in the adapter name. Full breakdown in
  `models_g4it/training_run_logs/20260525-085002_*.json`.
- Label partition: L_pos=120, L_neg=114, U=2049 (2283 items); 5110 sampled training pairs
  across 4 shapes (case_A / mixed_neg / mixed_pos / both_U). Seed 42.
- Ran **2026-05-25 08:39:46 → 15:19:17 CT**, exit 0 (`logs/run_s2.log`).

### Environment (from `logs/bootstrap_s2.log`)
- Pod image `runpod/pytorch:2.4.0-py3.11` (Python 3.11); repo checked out at commit `1bf002b7`.
- venv `--system-site-packages` + `requirements-gemma4.txt`; the install pulled **torch 2.12.0**
  (the pinned-deps + transitive torch), transformers 5.8.1, tokenizers 0.22.2,
  huggingface_hub 1.15.0, peft 0.14.0. (The torchaudio/torchvision "incompatible" warnings are
  benign — those libs are unused for this run.)

---

## How it was evaluated

Only **epoch1** was evaluated (epoch1 was the intended final RankAlign result; there was no time
to eval epoch2). Eval ran over all **82** humaneval-v2.1correct-upper problems in four scoring
modes, on separate eval pods (`cu-s2-eval-a/b/c/d`, now stopped):

| Mode | Column in result table | Status |
|------|------------------------|--------|
| self-typicality, no base | PMI self | 82/82 |
| neg-typicality, no base | Neg self | 82/82 |
| self-typicality + base | PMI base | 82/82 |
| neg-typicality + base | Neg base | **79/82** (problems `humaneval_1/10/100` fail in this mode only) |
| (raw logprob, in every CSV) | Raw | 82/82 |

- **Results:** `docs/humaneval_cu_v7_tables_2026-05-25.md`, the **"2 RankAlign"** row.
- The eval wrapper used on those pods was `setup_and_eval_s2ep1.sh` (not preserved — it lived on
  the eval pods). Its committed, parameterized descendant is
  **`scripts/setup_and_eval_s13ep0.sh`** (same logic; the s2 eval used the epoch1 adapter with
  `S=2, EP=1` and modes self/neg × bt0/bt1).

---

## Files in this directory

| Path | What it is |
|------|------------|
| `models_g4it/training_run_logs/20260525-085002_google--gemma-4-31B-it_humaneval-v2.1correct-upper.json` | Training provenance: delta config, shape-budget weights, label partition, sampled-pair counts, score stats, seed |
| `logs/train_s2.log` | Full training log (~1.25 MB) |
| `logs/run_s2.log`, `logs/run_s2_RankAlign.log` | Training-wrapper run logs — exact `TRAIN_FLAGS` + the (false) epoch2-missing FATAL |
| `logs/bootstrap_s2.log` | Pod environment setup log (repo commit, dep install) |
| `logs/hf_upload_ep01.log` | Confirms e0 + e1 auto-upload to latkes |
| `upload_cu_adapters.py` | Pod-side HF upload helper: maps each `/workspace/models_g4it/v7-*` folder → short repo name `rankalign-v7-g4-31b-{variant}-e{N}-cu-all`; `HF_ORG=latkes` |

---

## Reproduce, end to end

1. **Train** — on a gemma-4-capable pod (torch ≥ 2.5, Python ≥ 3.11), run the repo's RankAlign
   training script with the exact flags above (`--num_epochs 3` → e0/e1/e2).
2. **Upload** — `HF_TOKEN=… python upload_cu_adapters.py` (or upload each adapter folder).
3. **Eval** — run the eval (logic in `scripts/setup_and_eval_s13ep0.sh`) on the **epoch1** adapter
   over all 82 humaneval-v2.1correct-upper problems, modes self/neg × bt0/bt1.
4. **Build tables** — `python scripts/_build_humaneval_cu_v7_table.py` (per `HUMANEVAL_METRIC`)
   then `python scripts/_save_cu_v7_tables_md.py`.
