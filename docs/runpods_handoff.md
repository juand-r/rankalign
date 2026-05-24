# RunPods Handoff: Train + Eval Pipeline for s1-s12 × Models × Datasets

Self-contained spec for running the rankalign train+eval matrix on RunPods
(or any non-slurm GPU cloud). Written 2026-05-24 because our local slurm
cluster is contended and we may need a Plan B.

This doc is intended for a fresh Claude / engineer to read end-to-end and
execute. **Read everything before submitting work.** Special caveats are
called out inline.

---

## 1. Goal

For each cell `(setting, model, train_dataset)`:

1. **Train** a model with `scripts/ranking_loss_ref_fix.py` using the
   setting-specific flag combo.
2. **Evaluate** the trained model on the matching test tasks with
   `scripts/eval_by_claude.py`, producing per-task `scores_*.csv` files.

The matrix:

| Settings (priority order) | Models | Train dataset → Eval tasks |
|---|---|---|
| **High**: s4, s7, s2, s3, s1 | gemma-2-2b-it, gemma-2-9b-it | persona-v1 → 6 persona-v1-* tasks |
| **Lower**: s5, s6, s11, s12 | same | membership-sans-rosch-v0 → 10 rosch-* tasks |
| | | ifeval-concat → 21 ifeval-prompt_1..21 tasks |

**OUT OF SCOPE for this handoff**: ifeval-concat × gemma-2-9b-it
(infeasible runtime, see §11).

That gives **9 settings × 2 models × 3 datasets − 9 (skipped ifeval×9b-it) = 45 cells**.

---

## 2. Repo + branch

Github: `https://github.com/juand-r/rankalign` branch **`longform`**.

The `_overnight_launch.sh` dispatcher at HEAD encodes everything we
finalized this weekend. Commits to look for:
- `a711a1cb` — WALLTIME env override + 9h relaunch
- `fea681ef` — bumped walltimes + loosened `epoch[012]` glob
- `d50cf2c4` — initial overnight launchers
- `0d86c43`  — per-item disc-forward perf gating

If you make further changes, branch off `longform` to keep history clean.

---

## 3. Pod / environment setup

### 3.1 Recommended pod spec

For each of the 45 cells:

| Cell type | GPUs | RAM | Disk |
|---|---|---|---|
| 2b-it × any dataset | 1× A40 / A100 / H100 (24GB+ VRAM) | 64GB | 200GB scratch |
| 9b-it × persona/membership (LoRA) | 1× A40 / A100 / H100 (40GB+ preferred) | 96GB | 200GB scratch |

**ifeval-concat × 9b-it skipped** — see §11.

### 3.2 Environment install

Adapt the existing
[`setup-runpod-gemma4.sh`](../setup-runpod-gemma4.sh) (which we use for
Gemma-4 work). For these gemma-2 runs the standard `requirements.txt`
suffices. Minimal pod bootstrap:

```bash
# On a fresh pod (assumes torch is already in the base image with CUDA)
git clone -b longform --depth 1 https://github.com/juand-r/rankalign.git /workspace/rankalign
cd /workspace/rankalign

# venv inheriting torch from the pod image
python3 -m venv /workspace/.venv --system-site-packages
source /workspace/.venv/bin/activate
pip install --upgrade pip
# torch is NOT in requirements.txt; comes from base image.
pip install --ignore-installed -r requirements.txt

# HF caching to scratch (don't fill the root partition)
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $HF_HUB_CACHE /workspace/models2 /workspace/outputs

# HF token for gated gemma-2 models (you'll need a token with access)
export HF_TOKEN=hf_...
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# CUDA env that helps memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Sanity-check imports:
```bash
python -c "import torch, transformers, peft, trl; print(torch.cuda.is_available(), transformers.__version__)"
```

Should print `True 4.46.2` (or close).

### 3.3 Pre-download models

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('google/gemma-2-2b-it'); \
  snapshot_download('google/gemma-2-9b-it')"
```

---

## 4. Data files (already in the repo)

The repo already includes the data we need:

```
data/persona/v1/persona-{psychopathy,machiavellianism,narcissism,desire-to-create-allies,interest-in-music,interest-in-science}.csv
data/fixed-prompts-ifeval/gpt_ifeval_results_prompt_*.jsonl   # 99 files; per-prompt
data/...                                                       # rosch, etc, registered via src/tasks/*.py
```

Tasks are auto-registered via `import tasks` in `eval_by_claude.py` /
`ranking_loss_ref_fix.py`. **No extra data download needed** beyond
cloning the repo.

---

## 5. The 12 settings (canonical table from `docs/IMPORTANT-RESEARCH-PLAN.md`)

| # | Setting | Loss | Semi/lo | log-odds | force-same-x | TC |
|---|---|---|---|---|---|---|
| 1 | SFT-lo | sft | labelonly | — | — | — |
| 2 | RankAlign | pref-only | semi | — | — | — |
| 3 | New+fsx | comb | semi | ✓ | ✓ | — |
| 4 | New+fsx+tc | comb | semi | ✓ | ✓ | self |
| 5 | RankAlign+fsx+tc | pref-only | semi | — (NEW: was bug) | ✓ | self |
| 6 | RankAlign+tc | pref-only | semi | — | — | self |
| 7 | New+fsx+negtc | comb | semi | ✓ | ✓ | neg |
| 11 | New+tc | comb | semi | ✓ | — | self |
| 12 | New+negtc | comb | semi | ✓ | — | neg |

**For all `force-same-x` settings (s3, s4, s5, s7) we ALSO pass:**
- `--per-prompt-delta`
- `--shape-budget-mode global`

These are new flags introduced this week; defaults preserve old behavior.

**For all settings:**
- `--delta-bins 10` (auto-compute delta as `(p95−p5)/10`)
- `--disc-shots few` (NOT zero)
- `--num_epochs 3` (default)
- `--all`, `--split_type random`, `--alpha 1.0`, `--total_samples 5110` (default)

---

## 6. Train command (verbatim, from the dispatcher)

The dispatcher chains `_overnight_launch.sh` → `run_train_semi.sh` →
`python ranking_loss_ref_fix.py`. The actual python invocation looks like
this (with example flags for s4 × 9b-it × membership):

```bash
cd /workspace/rankalign/scripts
python ranking_loss_ref_fix.py \
    --model google/gemma-2-9b-it \
    --num_epochs 3 \
    --task membership-sans-rosch-v0 \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 1 \
    --all \
    --delta 0.15 \
    --force-same-x \
    --self-typicality \
    --validator-log-odds \
    --semi-supervised 0.1 \
    --disc-shots few \
    --lora \
    --models-dir /workspace/models2 \
    --delta-bins 10 \
    --per-prompt-delta \
    --shape-budget-mode global \
    --no-upload-hf \
    --no-wandb
```

Per-setting flag map (substitute into the template above):

| Setting | `--LOSS-flags` | `--force-same-x` | `--*-typicality` | `--validator-log-odds` | `--semi-supervised`/`--labeled-only` | extra |
|---|---|---|---|---|---|---|
| s1 | `--preference_loss_weight 0 --nll_validator_weight 1 --nll_generator_weight 1` | (omit) | (omit) | (omit) | `--labeled-only 0.1` | |
| s2 | `--preference_loss_weight 1 --nll_validator_weight 0 --nll_generator_weight 0` | (omit) | (omit) | (omit) | `--semi-supervised 0.1` | |
| s3 | `--preference_loss_weight 1 --nll_validator_weight 1 --nll_generator_weight 1` | `--force-same-x` | (omit) | `--validator-log-odds` | `--semi-supervised 0.1` | `--per-prompt-delta --shape-budget-mode global` |
| s4 | (same as s3) | `--force-same-x` | `--self-typicality` | `--validator-log-odds` | `--semi-supervised 0.1` | (same) |
| s5 | `--preference_loss_weight 1 --nll_validator_weight 0 --nll_generator_weight 0` | `--force-same-x` | `--self-typicality` | (omit; was a bug) | `--semi-supervised 0.1` | (same) |
| s6 | (same as s5) | (omit) | `--self-typicality` | (omit) | `--semi-supervised 0.1` | |
| s7 | (same as s3 weights) | `--force-same-x` | `--neg-typicality` | `--validator-log-odds` | `--semi-supervised 0.1` | (same as s3) |
| s11 | (same as s3 weights) | (omit) | `--self-typicality` | `--validator-log-odds` | `--semi-supervised 0.1` | |
| s12 | (same as s3 weights) | (omit) | `--neg-typicality` | `--validator-log-odds` | `--semi-supervised 0.1` | |

**LoRA flag**: pass `--lora` for any model whose name does NOT contain
`-2b` (i.e. for `gemma-2-9b-it` yes, for `gemma-2-2b-it` no — `-2b-it`
contains `-2b` so it's full-FT). This matches `run_train_semi.sh`'s
historical behavior.

**Per-dataset extras**:
- `ifeval-concat`: pass `--max-seq-len 1024` (long prompts).
- `persona-v1` and `membership-sans-rosch-v0`: no max-seq-len needed.

---

## 7. Output paths

The python script saves to `<MODELS_DIR>/v7-<MODEL>-delta<X.XX>-epoch<N>--<TASK>-all--d2g--random--alpha1.0<TC>--full-completion<PREF><NLLV><NLLG><FSX><PPD><VLO><SEMI>--fix1[_merged]`.

For LoRA runs there's also a `_merged` directory containing the merged
full model (used for eval).

Concrete example dir names per setting (substitute `${X.XX}` with the
auto-computed delta and `${EPOCH}` with 0/1/2):

```
# s1, gemma-2-2b-it, persona-v1
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1--fix1

# s2
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1

# s3
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1

# s4
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1

# s5
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--tc-self--full-completion--force-same-x--ppd--semi0.1--fix1

# s6
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--tc-self--full-completion--semi0.1--fix1

# s7
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1

# s11
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1

# s12
v7-google--gemma-2-2b-it-delta${X.XX}-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1
```

For 9b-it append `_merged` to the dir name when feeding to eval (the
LoRA-merged model is what eval reads).

---

## 8. Eval command

After train completes, run eval per (model_dir, task, TC variant). The
existing wrapper is `scripts/run_eval_semi.sh`; the underlying python is:

```bash
cd /workspace/rankalign/scripts
python eval_by_claude.py \
    --model "$MODEL_DIR" \
    --task "$TASK" \
    --split_type random \
    --disc-shots few \
    --gen-shots zero \
    --outputs-dir /workspace/outputs \
    --validator-log-odds \
    --self-typicality \
    --base-typicality \
    --base-model-name google/gemma-2-9b-it \
    --save-scores-csv
```

Eval task lists per training dataset:

```python
EVAL_TASKS = {
    "persona-v1": [
        "persona-v1-psychopathy", "persona-v1-machiavellianism",
        "persona-v1-narcissism", "persona-v1-desire-to-create-allies",
        "persona-v1-interest-in-music", "persona-v1-interest-in-science",
    ],
    "membership-sans-rosch-v0": [
        "rosch-bird", "rosch-carpenters-tool", "rosch-clothing",
        "rosch-fruit", "rosch-furniture", "rosch-sport", "rosch-toy",
        "rosch-vehicle", "rosch-vegetable", "rosch-weapon",
    ],
    "ifeval-concat": [f"ifeval-prompt_{i}" for i in range(1, 22)],  # 21 test-only
}
```

Per-setting TC variant for eval:

| Setting | Eval TC variant |
|---|---|
| s1, s2, s3 | `--self-typicality` (also do `--neg-typicality` if time) |
| s4, s5, s6, s11 | `--self-typicality` |
| s7, s12 | `--neg-typicality` |

**Always include** `--base-typicality --base-model-name <BASE_MODEL>`
where `<BASE_MODEL>` is the original HF id (e.g. `google/gemma-2-9b-it`).
This produces the matched `basetyp-` / `basetypneg-` CSV variants that
IRP §3 says we need.

Each eval emits a CSV under `outputs/scores_<prefix><model>_<task>_test_<metric>_<tc>_<lenorm>_<eos>_<ts>.csv`. The wrapper script
auto-skips a task if the CSV already exists (so re-running is cheap and
idempotent).

---

## 9. Recommended execution flow on RunPods

You have flexibility — the scripts work standalone. Two patterns:

### Pattern A: one pod per cell (simple, parallelizable)

For each of the 45 cells, spin up a fresh pod, run train + eval
sequentially in the same container, then sync outputs out and shut it
down.

Pros: clean isolation, easy retry logic.
Cons: pod startup overhead × 45 = significant.

### Pattern B: long-lived pods, multiple cells per pod (more efficient)

Spin up N pods. Each pod loops through a queue of cells, running
train→eval→sync→next.

Pros: lower overhead, better pod utilization.
Cons: a stuck cell holds up others; needs queue logic.

A simple pod-side runner:

```bash
#!/bin/bash
# pod_runner.sh — runs one cell end-to-end
DATASET=$1   # persona | membership | ifeval
MODEL=$2     # google/gemma-2-2b-it | google/gemma-2-9b-it
SETTING=$3   # s1..s7, s11, s12

set -e
source /workspace/.venv/bin/activate
cd /workspace/rankalign/scripts
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (run the training python invocation for this cell — substitute flags)
# ... see §6 ...
python ranking_loss_ref_fix.py ... \
    --models-dir /workspace/models2

# Find the highest-numbered epoch directory that was saved
MERGED_SUFFIX=""
[[ "$MODEL" != *"-2b"* ]] && MERGED_SUFFIX="_merged"
GLOB="/workspace/models2/v7-$(echo $MODEL | sed 's|/|--|g')-delta*-epoch[012]--<TASK>-all--d2g--random--alpha1.0${TC_LABEL}--full-completion${PREF}${NLLV}${NLLG}${FSX}${PPD}${VLO}${SEMI}--fix1${MERGED_SUFFIX}"
MODEL_DIR=$(ls -dt $GLOB 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "no model saved"; exit 1; }

# Eval each test task
for TASK in <eval task list>; do
    bash run_eval_semi.sh "$MODEL_DIR" \
        --self-typcorr --base-typcorr --base-model "$MODEL" --log-odds \
        --outputs-dir /workspace/outputs \
        -- "$TASK"
done

# Sync outputs out (s3, gdrive, scp — your choice)
# rsync -av /workspace/outputs/ s3://...
```

---

## 10. Recovery / failure modes

The `_overnight_launch.sh` we use locally:
- Uses `epoch[012]` glob so a walltime-killed run that only saved
  `epoch0` or `epoch1` is still evaluable. **Replicate this on
  RunPods.** Check `ls -dt $GLOB | head -1` to pick the most recent
  saved epoch.
- Uses `afterany` (not `afterok`) for eval dependency. On RunPods,
  just run eval after train returns regardless of exit code, then
  fall back to `if [ -d ... ]` checks.

Common failure modes seen this week:

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: [Errno 28] No space left on device` | `--models-dir` resolved to a tiny disk partition | Use absolute path under `/workspace/models2` (or wherever scratch is). |
| `argparse: % must be escaped` | `argparse` `%` in help string | Already fixed; but if a new flag is added with `%` in help, escape as `%%`. |
| Per-prompt-delta + delta_bins=None | flag sanity check | Always pass both `--per-prompt-delta` AND `--delta-bins 10` together. |
| Disc-shots few but task expects zero | Some tasks (codecontests, humaneval) are zero-only | Out of scope for this handoff. |

---

## 11. The skipped cell: ifeval-concat × gemma-2-9b-it

Estimated runtime: ~40-50h for 3 epochs at observed ~6s/iter (LoRA, long
seqs). Even with `--max-seq-len 1024` and 2-GPU model parallel, a single
cell exceeds typical pod walltime budgets and slurm `bf_window`. The
local launcher script SKIPs this combination.

If you must run it on RunPods:
- Use 2× A100/H100 (40-80GB each)
- Pass `--gradient_checkpointing` (memory) and consider `--num_epochs 1`
  (runtime).
- Allow ~24h walltime.

Otherwise, leave this cell incomplete; per IRP this is a tertiary task.

---

## 12. Final deliverables back to me

After all 45 cells finish:

1. `outputs/scores_*.csv` files (per-task, per-tc-variant). These can be
   uploaded to `/datastor1/jdr/gv-gap/rankalign/outputs/` on the local
   cluster (the `metrics-from-scores/*.py` analysis scripts read from
   there).
2. `models2/training_run_logs/*.json` — per-run JSON with stratified
   sampling stats, delta, p5/p95, etc. Useful for diagnostics.
3. The trained model dirs themselves are large; not strictly needed if
   the eval CSVs are reproducible. Up to you whether to upload them.

---

## 13. References (read these for full context)

- [`docs/IMPORTANT-RESEARCH-PLAN.md`](IMPORTANT-RESEARCH-PLAN.md) — the canonical
  spec for s1-s12, eval rules, and what "beating RankAlign" means.
- [`docs/overnight_run_instructions.md`](overnight_run_instructions.md) — what
  the user explicitly asked me to do this weekend (verbatim).
- [`scripts/_overnight_launch.sh`](../scripts/_overnight_launch.sh) — the
  authoritative source of truth for flag combinations per setting. If
  there's any conflict between this doc and the script, **trust the
  script**.
- [`scripts/run_train_semi.sh`](../scripts/run_train_semi.sh) — the original
  train wrapper that `_overnight_launch.sh` calls.
- [`scripts/ranking_loss_ref_fix.py`](../scripts/ranking_loss_ref_fix.py) — the
  fix1 training script. The arg-parser section (~line 60-200) is the
  ultimate source of truth for what flags do.
- [`scripts/eval_by_claude.py`](../scripts/eval_by_claude.py) — the eval
  script. Arg-parser is at the top.
