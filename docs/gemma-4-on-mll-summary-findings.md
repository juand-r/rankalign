# Gemma-4-31B-it on mll (Slurm) — summary of findings

What's in the repo for training and evaluating `google/gemma-4-31B-it`
(LoRA) on the **mll** Slurm cluster (UT Austin CS, Durrett group). This is
a navigation doc — pointers to the files that do the work, plus the key
context for using them. The authoritative setup doc is
[`mll_gemma4_training_setup.md`](mll_gemma4_training_setup.md); this file
is a higher-level index over **everything** related to the mll gemma-4
workflow.

Status (per the setup doc): **infrastructure ready, no training launched
yet.** Launching is a "big job" — confirm before submitting.

---

## 1. The three files that actually run training on mll

The whole mll workflow is three committed files; no ad-hoc commands
produced any artifact.

| Path | Role |
|---|---|
| [`setup-mll-gemma4.sh`](../setup-mll-gemma4.sh) | One-shot venv builder. Login-node-safe (no GPU assertion). |
| [`pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch`](../pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch) | Slurm wrapper — sets mll paths, requests GPUs, execs the shared entry point. |
| [`pod-setup-train-scripts-gemma-4/run_arm_3epoch.sh`](../pod-setup-train-scripts-gemma-4/run_arm_3epoch.sh) | **Single** train+eval entry point shared with the RunPod pod. Parameterized by env vars. |

The sbatch wrapper is a thin shim: it sets `RANKALIGN_DIR`, `VENV_DIR`,
`MODELS_DIR`, `OUTDIR`, `LOGDIR`, `HF_HOME` to mll values and then
`exec bash`'s `run_arm_3epoch.sh`. Defaults inside `run_arm_3epoch.sh`
are the RunPod `/workspace` layout, so the same script runs byte-identically
on the pod and on mll (DRY — one source of train/eval truth).

There is also a base-model eval wrapper:

| Path | Role |
|---|---|
| [`pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch`](../pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch) | Evaluates the untrained base `google/gemma-4-31B-it` (self-/neg-TC) on humaneval-v2.x tasks. Runs on `allnodes` (better queue times) since no training is involved. |

## 2. The training code

[`scripts/ranking_loss_ref_gemma4.py`](../scripts/ranking_loss_ref_gemma4.py)
— the gemma-4 branch of the ranking-loss trainer (~3190 lines). Invoked
by `run_arm_3epoch.sh` with this fixed `COMMON` flag set:

```
--model google/gemma-4-31B-it --num_epochs 3 --task <HE_TASK>
--train_g_or_d g --split_type random --nll_validator_weight 0
--nll_generator_weight 0 --preference_loss_weight 1 --all --delta 0.15
--semi-supervised 0.1 --disc-shots zero --lora --gradient_checkpointing
--models-dir <MODELS_DIR> --total_samples 5110
```

Per-arm flag added on top:

| Arm | Train flag | Eval modes |
|---|---|---|
| `no_tc` (RankAlign #2 baseline) | — | `--self-typicality` **and** `--neg-typicality` |
| `self_tc` (RankAlign+tc #6) | `--self-typicality` | `--self-typicality` |
| `neg_tc` (RankAlign+negtc #9) | `--neg-typicality` | `--neg-typicality` |

Eval is run via `scripts/eval_by_claude.py --base-typicality
--base-model google/gemma-4-31B-it`, identical to the verified
`recovery_pipeline_v2.sh` STEP 2+3 from the pod era. Produces
`scores_basetyp-` / `scores_basetypneg-` CSVs.

## 3. mll-specific constraints (why setup looks the way it does)

From `docs/mll_gemma4_training_setup.md` §1:

| Constraint | Implication |
|---|---|
| mll GPUs are **NVIDIA A40, 48 GB**, 8/node | 31B model (~62 GB bf16) does **not** fit one GPU — needs `device_map="auto"` across ≥2–3 GPUs |
| mll system python is **3.8** | Too old for `transformers` git-main (gemma-4). venv must use **/lusr python 3.10** |
| `requirements.txt` pins `transformers==4.46.2` | Does **not** load gemma-4 — overridden with git-main in the venv |
| Login node has **no GPU** | `torch.cuda.is_available()` is asserted **in the Slurm job**, not at venv-build time |
| `$HOME` 20 GB quota | venv + HF cache + outputs all live on `/datastor2/jdr` |

## 4. On-mll paths (set up once)

| Thing | Path |
|---|---|
| venv | `/datastor2/jdr/venvs/gemma4` |
| rankalign clone | `/datastor1/jdr/gv-gap/rankalign` (branch `longform`) |
| HF cache | `/datastor2/jdr/.cache/huggingface` |
| HF token | `/datastor2/jdr/.hftoken` (mode 600, never committed) |
| Model adapters out | `/datastor2/jdr/models_g4it` |
| Score CSVs out (trained) | `/datastor2/jdr/outputs_gemma4_mll` |
| Score CSVs out (base eval) | `/datastor2/jdr/rankalign/outputs_gemma4_from_pod` (default in `mll_eval_base.sbatch`) |
| Logs | `/datastor2/jdr/logs` |

### venv contents (verified, per the setup doc)

python 3.10.16 · torch 2.5.1+cu124 (A40-compatible) · transformers
5.8.0.dev0 (git-main, gemma-4 support) · tokenizers 0.22.2 ·
huggingface_hub 1.15.0 · peft 0.14.0 · accelerate 1.1.0 · numpy 2.1.3 ·
scipy 1.14.1 · scikit-learn 1.5.2.

Rebuild any time (idempotent):

```bash
bash /datastor1/jdr/gv-gap/rankalign/setup-mll-gemma4.sh
```

## 5. Slurm configuration

### Training (`mll_train_arm.sbatch`)

```
#SBATCH --partition=p-gdurret      # Durrett group node, infinite timelimit, 1031 GB RAM
#SBATCH --account=gdurrett
#SBATCH --gres=gpu:3               # 3 × A40 = 144 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
```

`p-gdurret` is chosen (not `allnodes`) because `allnodes` has a ≤9 CPU /
≤96 GB cap that this job exceeds; `p-gdurret` does not. The setup doc
reports `sbatch --test-only` schedules immediately on a p-gdurret A40 node.

### Base eval (`mll_eval_base.sbatch`)

```
#SBATCH --partition=allnodes       # 11 nodes — much better queue times than p-gdurret's 1
#SBATCH --account=gdurrett
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
```

`allnodes` is fine here because base eval requests fit its caps (120 GB
< 96 GB cap? — note: the script requests 120 GB; the setup doc says the
cap is ≤96 GB. If `allnodes` rejects, the fallback is `p-gdurret` or
reducing `--mem`.) Base eval only loads one 62 GB model (no second
`--base-typicality` reference loaded simultaneously), so 3 × A40 has lots
of headroom.

### Is 3 GPUs enough?

- **Training**: yes, comfortably. 62 GB model + LoRA + gradient
  checkpointing shards across 3×A40 (144 GB).
- **Trained-model eval**: tight. `eval_by_claude.py --base-typicality`
  loads the trained model **and** a separate base model ≈ 124 GB.
  Fits 144 GB but with little headroom — **may OOM during eval**. The
  fix is a one-line bump in `mll_train_arm.sbatch`:
  `--gres=gpu:3` → `--gres=gpu:4`. Training is safe at 3; only eval
  carries the risk.
- **Base eval**: no second model is ever resident, so 3×A40 is plenty
  (2×A40 = 96 GB would also fit but is the floor).

## 6. How to launch

> Per `.cursor/rules/ask-before-launching-jobs.mdc`, do not submit any of
> these without an explicit "yes, launch it" from the user.

### Training (three arms, run concurrently)

```bash
cd /datastor1/jdr/gv-gap/rankalign
sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch no_tc
sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch self_tc
sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch neg_tc
```

Each is an independent job; they can run concurrently. Each: trains
3 epochs, then evals the final-epoch (`EVAL_EPOCHS=2`, 0-indexed)
checkpoint.

### Base-model eval

```bash
cd /datastor1/jdr/gv-gap/rankalign
sbatch pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch self_tc
sbatch pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch neg_tc
# or both in one job:
sbatch pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch both
```

Override dataset via `HE_TASK=humaneval-v2.1correct-multi sbatch ...`.
Override outdir via `OUTDIR=/path sbatch ...`.

### Behavior (both training and base eval)

- **Idempotent / resumable**: each `(epoch, mode, task)` (or
  `(model, mode, task)` for base eval) has a done-marker file under
  `$OUTDIR/.done/` or `$OUTDIR/.done_base/`. Re-submitting after a
  crash resumes — completed cells are skipped.
- **Training is NOT resumable mid-run.** A training crash retrains
  that arm from scratch. Run one arm per job for fault isolation.
- **Default eval scope**: only the final epoch (`EVAL_EPOCHS=2`). Override
  with `EVAL_EPOCHS="0 1 2"` to also eval intermediate epochs.
- **`BASETYP=1`** (default): typicality reference = frozen base model.
  Marker/log names include `bt<N>` so a `BASETYP=0` re-run does not
  collide with the `BASETYP=1` results.

## 7. Monitoring

```bash
tail -f /datastor2/jdr/logs/run_<arm>_3epoch.log     # training arm
tail -f /datastor2/jdr/logs/eval_base_<mode>_<task>.log  # base eval
squeue -u jdr
```

Note: per `.cursor/rules/slurm-job-monitoring.mdc`, do not rely on
stderr tqdm progress bars to gauge overall progress — they only show
the current epoch. Use the explicit `EPOCH<N>_EVAL_COMPLETE` /
`ARM_<arm>_COMPLETE` markers in the `.log` files, or
`grep "Epoch \[" ~/logs/<JOBID>.out` for epoch progress.

## 8. Analysis

Per the setup doc, analyze trained-model results with
`scripts-more/analyze_gemma4_tc.py` pointed at `--scores-dir
/datastor2/jdr/outputs_gemma4_mll`. The general humaneval table builder
[`scripts/_build_humaneval_table.py`](../scripts/_build_humaneval_table.py)
also scans these dirs.

## 9. Related docs (full index)

| Doc | Scope |
|---|---|
| [`mll_gemma4_training_setup.md`](mll_gemma4_training_setup.md) | **Authoritative** mll setup guide — constraints, paths, Slurm config, launch, monitoring |
| [`gemma4-31b-pod-setup.md`](gemma4-31b-pod-setup.md) | Original RunPod pod setup the mll infra mirrors |
| [`gemma4_v21correct_multi_trained_models.md`](gemma4_v21correct_multi_trained_models.md) | Provenance for the trained checkpoints on `v2.1correct-multi` |
| [`IMPORTANT-RESEARCH-PLAN.md`](IMPORTANT-RESEARCH-PLAN.md) | The priority-1 TC comparison (RankAlign #2 vs #6 vs #9) this run is meant to produce |
| [`../pod-setup-train-scripts-gemma-4/README.md`](../pod-setup-train-scripts-gemma-4/README.md) | Chronological provenance of the pod-era scripts (`sequential_pipeline.sh`, `real_pipeline_g4it.sh`, `recovery_pipeline_v2.sh`, …) — why `run_arm_3epoch.sh` ended up the way it did |

## 10. Reproducibility

Every step is a committed script — no ad-hoc commands produced any
artifact. `setup-mll-gemma4.sh` rebuilds the venv; `mll_train_arm.sbatch`
+ `run_arm_3epoch.sh` reproduce training/eval exactly; `mll_eval_base.sbatch`
reproduces base eval exactly. The HF token is the only out-of-band item
(gitignored file on scratch, mode 600; rotate when convenient).
