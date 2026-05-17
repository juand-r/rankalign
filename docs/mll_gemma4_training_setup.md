# Training gemma-4-31B-it (LoRA) on mll

Infrastructure for training + evaluating `google/gemma-4-31B-it` LoRA
variants on the **mll** Slurm cluster (UT Austin CS). Mirrors the RunPod
pod setup; the same committed train+eval entry point runs in both places.

Status: **infrastructure ready, no training launched.** You launch variants
with one `sbatch` command (see [Launching](#launching-a-variant)).

---

## 1. Why mll needs special setup

| Constraint | Implication |
|---|---|
| mll GPUs are **NVIDIA A40, 48 GB** each, 8/node | 31B model (~62 GB bf16) does **not** fit one GPU — needs `device_map="auto"` across ≥2–3 |
| mll system python is **3.8** | Too old for `transformers` git-main (gemma-4). venv must use **/lusr python 3.10** |
| `requirements.txt` pins `transformers==4.46.2` | Does **not** load gemma-4 — overridden with git-main in the venv |
| Login node has **no GPU** | `torch.cuda.is_available()` is asserted **in the Slurm job**, not at venv-build time |
| `$HOME` 20 GB quota | venv + HF cache + outputs all live on `/datastor2/jdr` |

---

## 2. Components (all committed on `longform`)

| Path | What |
|---|---|
| `setup-mll-gemma4.sh` | One-shot venv builder (repo root, beside `setup-runpod-gemma4.sh`) |
| `pod-setup-train-scripts-gemma-4/run_arm_3epoch.sh` | **Single** train+eval entry point — parameterized, used by **both** pod and mll |
| `pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch` | mll Slurm wrapper (sets mll paths, requests GPUs, execs `run_arm_3epoch.sh`) |

`run_arm_3epoch.sh` paths are env-overridable (`RANKALIGN_DIR`, `VENV_DIR`,
`MODELS_DIR`, `OUTDIR`, `LOGDIR`, `HF_HOME`); defaults are the RunPod
`/workspace` layout, so **existing pod usage is byte-identical** (DRY — one
source of train/eval truth). The sbatch wrapper sets the mll values.

### On-mll locations (set up once)

| Thing | Path |
|---|---|
| venv | `/datastor2/jdr/venvs/gemma4` |
| rankalign clone | `/datastor1/jdr/gv-gap/rankalign` (branch `longform`) |
| HF cache | `/datastor2/jdr/.cache/huggingface` |
| HF token | `/datastor2/jdr/.hftoken` (mode 600, never committed) |
| model adapters out | `/datastor2/jdr/models_g4it` |
| score CSVs out | `/datastor2/jdr/outputs_gemma4_mll` |
| logs | `/datastor2/jdr/logs` |

### venv contents (verified)

python 3.10.16 · torch 2.5.1+cu124 (A40-compatible) · transformers
5.8.0.dev0 (git-main, gemma-4 support) · tokenizers 0.22.2 ·
huggingface_hub 1.15.0 · peft 0.14.0 · accelerate 1.1.0 · numpy 2.1.3 ·
scipy 1.14.1 · scikit-learn 1.5.2.

Rebuild any time (idempotent): `bash /datastor1/jdr/gv-gap/rankalign/setup-mll-gemma4.sh`

---

## 3. Slurm configuration

`mll_train_arm.sbatch` requests:

```
#SBATCH --partition=p-gdurret      # Durrett group node, infinite timelimit, 1031 GB RAM
#SBATCH --account=gdurrett
#SBATCH --gres=gpu:3               # 3 × A40 = 144 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
```

Validated with `sbatch --test-only` → schedules immediately on a p-gdurret
A40 node. (The `allnodes` partition has a ≤9 CPU / ≤96 GB cap; `p-gdurret`
does not — it is the group's dedicated node.)

### Is 3 GPUs enough?

- **Training**: yes, comfortably. 62 GB model + LoRA + gradient
  checkpointing shards across 3×A40 (144 GB).
- **Eval**: tight. `eval_by_claude.py --base-typicality` loads the trained
  model **and** a separate base model ≈ 124 GB. Fits 144 GB but with
  little headroom — it **may OOM during eval**. If it does, the fix is one
  line in `mll_train_arm.sbatch`: `--gres=gpu:3` → `--gres=gpu:4`.
  Training is safe at 3; only eval carries the risk.

---

## 4. Launching a variant

Three arms (the priority-1 TC comparison — see
[`IMPORTANT-RESEARCH-PLAN.md`](IMPORTANT-RESEARCH-PLAN.md) §3):

| arm | train flag | eval modes |
|---|---|---|
| `no_tc` (RankAlign #2 baseline) | — | self-TC **and** neg-TC |
| `self_tc` (RankAlign+tc #6) | `--self-typicality` | self-TC |
| `neg_tc` (RankAlign+negtc #9) | `--neg-typicality` | neg-TC |

```bash
raca ssh mll 'cd /datastor1/jdr/gv-gap/rankalign && \
  sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch no_tc'
# repeat with self_tc / neg_tc — independent jobs, can run concurrently
```

Each job: trains `--num_epochs 3` then evals. **What it trains** (the
verified `COMMON`, identical to `real_pipeline_g4it.sh`):

```
--model google/gemma-4-31B-it --num_epochs 3 --task humaneval-v2.1correct-upper
--train_g_or_d g --split_type random --nll_validator_weight 0
--nll_generator_weight 0 --preference_loss_weight 1 --all --delta 0.15
--semi-supervised 0.1 --disc-shots zero --lora --gradient_checkpointing
--total_samples 5110
```

### Behavior

- **Idempotent / resumable**: training skipped if all 3 epoch adapters
  exist; each `(epoch, mode, task)` eval skipped if its done-marker exists.
  Re-submitting after a crash resumes (training itself is not resumable
  mid-run — a training crash retrains that arm from scratch; run one arm
  per job for fault isolation).
- **Eval scope**: by default only the **final epoch** (`EVAL_EPOCHS=2`,
  0-indexed = 3rd epoch — the lab-standard comparison point). Override with
  `EVAL_EPOCHS="0 1 2"` to also eval intermediate epochs.
- **Eval recipe**: `--base-typicality --base-model google/gemma-4-31B-it`
  + matching `--self/neg-typicality`, identical to the verified
  `recovery_pipeline_v2.sh` STEP 2+3. Produces `scores_basetyp-` /
  `scores_basetypneg-` CSVs in `/datastor2/jdr/outputs_gemma4_mll`.

### Monitoring

```bash
raca ssh mll 'tail -f /datastor2/jdr/logs/run_<arm>_3epoch.log'
raca ssh mll 'squeue -u jdr'
```

Analyze results with `scripts-more/analyze_gemma4_tc.py` (point
`--scores-dir` at `/datastor2/jdr/outputs_gemma4_mll`).

---

## 5. Reproducibility

Every step is a committed script — no ad-hoc commands produced any
artifact. `setup-mll-gemma4.sh` rebuilds the venv; `mll_train_arm.sbatch`
+ `run_arm_3epoch.sh` reproduce training/eval exactly. The HF token is the
only out-of-band item (gitignored file on scratch, reused per prior
decision; rotate when convenient).
