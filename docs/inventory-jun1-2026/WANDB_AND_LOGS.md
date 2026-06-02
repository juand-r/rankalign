# WandB coverage + training-JSON-log structure

**Compiled 2026-06-02.** Two questions: (1) which experiments/settings have wandb logs and
which don't; (2) what the on-disk JSON training logs look like.

Raw data: `_raw/wandb_runs.jsonl` (103 local wandb runs, args-parsed),
`_raw/training_logs_flags.jsonl` (85 JSON logs). Regenerate via `_dump_wandb_runs.py` /
`_dump_training_flags.py` on mll.

---

## 1. WandB

### How it's wired
- Trainer `scripts/ranking_loss_ref_fix.py`: **wandb ON by default**, `project="rankalign"`,
  auto-generated run name; disabled only with `--no-wandb`. Mode follows the `WANDB_MODE` env var.
- **mll is logged in to wandb** (`~/.netrc` has `api.wandb` creds) → online runs there sync to
  the `rankalign` project in the cloud.

### Per-launcher (this determines coverage)
| Launcher | used for | wandb? | where the logs are |
|----------|----------|--------|--------------------|
| `_overnight_launch.sh` → `run_train_semi.sh` (mll Slurm) | gemma-2 persona/membership/ifeval | **ON, online** (no `--no-wandb`, no WANDB_MODE) | cloud `rankalign` project **+** local dirs `/datastor2/jdr/rankalign/{scripts/,}wandb/run-*` |
| `run_settings_v21correct_{upper,multi}.sh` (gemma-4 pods) | gemma-4-31B-it cu/cm | **ON, but `WANDB_MODE=offline`** | **offline** run dirs **on the pods** (now stopped) — NOT synced, NOT on mll → effectively lost |
| `run_gemma2_cell.sh`, `run_qwen35_cell.sh`, `run_gemma2_v7b_cell.sh`, `run_qwen35_v7b_cell.sh` (pods) | gemma-2-9b-it (ra9b) + qwen3.5-9b, **all v7b** | **`--no-wandb` → NONE** | — |

### ⚠ The CLOUD is authoritative — local dirs badly undercount
The runs that train on **mll** log **online** and sync to the cloud project
**`juand-r/rankalign`**; their *local* `wandb/run-*` dirs were mostly cleaned afterward, so the
103 local dirs (all persona, May 19–23) are NOT a complete picture. **You must query the cloud.**
Authoritative count from the wandb API (`_raw/wandb_cloud_runs.json`, 2026-06-02): **1339 runs**
in `juand-r/rankalign`. For the three paper datasets:

| Paper dataset (wandb `task`) | wandb in cloud? | runs by model |
|------------------------------|-----------------|---------------|
| **ifeval** (`ifeval-concat`) | ✅ **YES** | gemma-2-2b **70**, gemma-2-2b-it **13**, gemma-2-9b-it **36**, Qwen3.5-9B **4** (123 total; gemma-2-2b spans Mar–May, 9b-it Apr–May) |
| **rosch / hyponymy** (`membership-sans-rosch-v0`) | ✅ **YES** | gemma-2-2b **42**, gemma-2-2b-it **17**, gemma-2-9b-it **26**, Qwen3.5-9B **1** (86 total, all May; + `rosch-furniture-and-bird` 19) |
| **humaneval-cu** (`humaneval-v2.1correct-upper`) | ❌ **essentially NO** | gemma-4-31B-it: **1** run only |

(The cloud also holds the project's full history — persona-v1 123, plausibleqa 113, ambigqa 74,
hypernym-* hundreds, etc. — not paper-relevant here.)

### Bottom line — who has wandb and who doesn't
- ✅ **HAS wandb (cloud `juand-r/rankalign`):** **ifeval** and **rosch/membership** for all three
  gemma-2 models (+ a little qwen3.5-9b), and **persona-v1** (all gemma-2). These were trained on
  **mll** (online wandb).
- ❌ **The one real gap is humaneval-cu (gemma-4-31B-it): only 1 cloud run.** The gemma-4 pods ran
  `WANDB_MODE=offline`, so their curves sat on the (now-stopped) pods and never synced. humaneval-cm
  likewise has no gemma-4 cloud wandb. Treat gemma-4 training curves as lost unless a pod volume is revived.
- ⚠ **Coverage is per-*run*, not cleanly per-final-setting.** 1339 runs include many sweeps /
  re-runs / exploratory configs. "ifeval/rosch have wandb" means curves exist in the project; mapping
  each *final paper model* to its specific cloud run needs a name/config match (run names encode
  `model-task-…-delta…-nllv…-pref…-semi…`). I can build that map if you want it.

### Per-final-model run map (the precision view) → `WANDB_RUN_MAP.md`
"Task has wandb" hides important per-cell gaps. The full map (every paper
model×setting → its wandb run name + URL + state) is in **`WANDB_RUN_MAP.md`**, built from
`_raw/wandb_paper_runs.json` (210 paper-task runs: **82 finished, 64 failed, 64 crashed**).
Highlights of what's actually *finished* (= complete curves) for paper cells:
- **rosch/membership** — best covered: gemma-2-2b-it and gemma-2-9b-it have a **finished
  delta-bins run for every setting** (s1–s13), dated May 24–25. gemma-2-2b mostly finished too.
- **ifeval** — patchier: gemma-2-2b has finished runs (but mostly *early fixed-delta*, Mar–Apr);
  **gemma-2-2b-it ifeval s1–s7 are all `crashed` (0 finished)** — only its s13 finished;
  gemma-2-9b-it has s1/s2/s7/s8 finished but **s13 failed**.
- **humaneval-cu** — only one gemma-4 run reached the cloud and it **`crashed`** (0 finished). No
  usable curves. (Matches: gemma-4 pods were offline.)
- Qwen3.5-9B: only a couple of finished s2 runs (ifeval s2 failed).

So even for the tasks that "have wandb," not every final model has a finished run — use
`WANDB_RUN_MAP.md` to see exactly which do.

### Correction note
An earlier version of this doc said membership/ifeval had "no wandb" — that was based on **local
dirs only** and was **wrong**. The cloud query (above) shows ifeval + rosch/membership are well
covered online; only **humaneval-cu (gemma-4)** is genuinely missing.

---

## 2. Training JSON logs

### Where
`<models-dir>/training_run_logs/<timestamp>_google--<model>_<task>.json`, written by the trainer
itself. **85 logs** at `/datastor2/jdr/rankalign/models2/training_run_logs/` — these cover the
**May 24–25 gemma-2 batch** (gemma-2-2b / 2b-it / 9b-it × persona/membership/ifeval) **+ 1**
gemma-4-31B-it cu **s13**. (Setting→file map: `v7/training_logs_map_v7.md`.)

**Coverage is complementary to wandb:** the JSON logs exist for exactly the runs that *lack*
wandb (the May 24–25 batch). Gaps:
- **gemma-4 cu s2** JSON is preserved in `../provenance-cu-s2-rankalign/models_g4it/training_run_logs/`;
  **cu s3/s4/s7** JSONs were on the (stopped) cu pods → `[PENDING — pod volume]`.
- **qwen3.5-9b** has **no** JSON logs on mll (trained on pods; its `training_run_logs/` was on the
  pod volume) → `[PENDING — pod volume]`.

### What a JSON log looks like (full schema)
Top-level keys: `timestamp, model, task, flags, [consistency_ft], shape_weights,
per_shape_budget, score_metric, n_items_total, global_score_stats, label_partition,
valid_pairs_by_shape, sampled_by_shape, total_valid_pairs, total_sampled, n_prompt_groups,
per_prompt[]`.

Example — `20260524-125542_google--gemma-2-9b-it_membership-sans-rosch-v0.json` (a comb/New+fsx run):
```json
{
  "timestamp": "2026-05-24 12:55:42",
  "model": "google/gemma-2-9b-it",
  "task": "membership-sans-rosch-v0",
  "flags": {                          // the run's training args (used for setting classification)
    "force_same_x": true, "per_prompt_delta": true, "shape_budget_mode": "global",
    "delta_bins": 10, "delta_arg": 0.15, "validator_log_odds": true,
    "self_typicality": false, "neg_typicality": false,
    "semi_supervised": 0.1, "labeled_only": null, "split_seed": 42
  },
  "shape_weights": {"case_A":0.2,"mixed_neg":0.2,"mixed_pos":0.2,"both_U":0.4},
  "per_shape_budget": {"case_A":1186,"both_U":3924},
  "score_metric": "validator log-odds",
  "n_items_total": 2068,
  "global_score_stats": {             // p5/p95 spread -> the realized delta (delta_used)
    "min":-15.94,"p5":-14.05,"p95":12.87,"max":15.29,
    "spread_p5_p95":26.91,"delta_used":2.691            // <- this is the "delta2.69" in the dir name
  },
  "label_partition": {"L_pos":85,"L_neg":88,"U":1895},  // labeled-pos/neg + unlabeled counts
  "valid_pairs_by_shape": {"case_A":1186,"mixed_neg":0,"mixed_pos":0,"both_U":21370},
  "sampled_by_shape":     {"case_A":1186,"mixed_neg":0,"mixed_pos":0,"both_U":3924},
  "total_valid_pairs": 22556, "total_sampled": 5110,    // pairs available vs actually trained on
  "n_prompt_groups": 68,
  "per_prompt": [ { "n_items":96, "p5":-11.81, "p95":11.46, "spread_p5_p95":23.27,
                    "delta_used":2.327, "prompt":"Complete the sentence: ...",
                    "valid_pairs_by_shape":{...}, "n_sampled":678 }, ... ]  // one entry per prompt
}
```

**s13 (consistency-ft) adds a `consistency_ft` block** recording the filter, e.g. from the
gemma-4 cu s13 log (`20260524-193331_…humaneval-v2.1correct-upper.json`):
```json
"consistency_ft": {
  "t_v": -11.20, "t_g": -128.22, "threshold_basis": "all-items",
  "n_total_pre_filter": 234, "n_labeled_total": 234,
  "n_labeled_kept": 155, "n_labeled_dropped": 79, "n_unlabeled_kept": 0, "n_kept": 155,
  "label_cells_bv_bg": {"00_low_low":61,"01_low_high":60,"10_high_low":19,"11_high_high":94}
}
```
(`bv`=binarized validator score, `bg`=binarized generator score; s13 drops the off-diagonal
`01`/`10` labeled items where the two disagree — here 60+19=79 dropped, matching `n_labeled_dropped`.)

### What the JSON does NOT contain
No loss curves / per-step metrics / final eval scores (those are in wandb for persona, or in the
`scores_*.csv` for evals). The JSON is a **data-construction provenance record**: exact flags,
realized delta, label partition, and how many pairs of each shape were sampled. To read one:
`jq . <file>` on mll.

---

## 3. What each wandb run actually stores

Verified by inspecting live finished runs via the API (`_dump_wandb_contents.py`). Every run in
`juand-r/rankalign` holds four things: **config**, **logged metric time-series**, a **summary**,
and **attached files** (incl. a full code snapshot).

### (a) config — hyperparameters/args (28 keys)
`model, task, train_g_or_d, split_type, split_seed, delta, delta_bins, alpha, num_epochs,
learning_rate, total_samples, preference_loss_weight, nll_validator_weight,
nll_generator_weight, self_typicality, neg_typicality, typicality_correction, semi_supervised,
labeled_only, use_all, use_lora, use_full_completion, single_token_data_only, with_ref,
gradient_checkpointing,` **plus the delta-bins stats added after init:**
`validator_score_min, validator_score_max, validator_score_spread_p5_p95`.

> ⚠ config does **NOT** include `force_same_x`, `validator_log_odds`, `consistency_ft`, or
> `per_prompt_delta`. Those live only in `wandb-metadata.json`'s `args` (see (d)) — which is why
> the setting classification in `WANDB_RUN_MAP.md` reads the args, not the config.

### (b) logged metrics — the training time-series (`run.history`, ~1 row per optimizer step; this run had _step≈15.3k)
Per **step** (`train/`): `loss, preference_loss, nll_validator_loss, nll_generator_loss,
score_i, score_j, diff, epoch, global_step`.
Only when **NLL-generator is on** (comb settings, `nll_generator_weight>0`) it additionally logs:
`score_gen_i, score_gen_j, indicator_i, indicator_j`. (Pref-only / SFT runs lack these 4.)
Per **epoch** (`epoch/`): `avg_loss, epoch`.
wandb internals on every row: `_step, _runtime, _timestamp`. wandb also auto-captures **system
metrics** (GPU/CPU/mem util, `system/*`) viewable in the UI.

> What's **NOT** here: **no eval metrics** (no gen_roc / val_roc / pearson). wandb is
> **training-dynamics only** — loss + score margins. Eval numbers live in the `scores_*.csv` /
> `metrics-from-scores/` (TASK 2). So wandb answers "did training converge / was the loss healthy",
> not "how did the model score".

### (c) summary — final value of each logged metric
The last-step value of every `train/*` and `epoch/*` key, plus `_runtime` (wall-seconds),
`_step`, `_timestamp`, `_wandb`. (Note: `train/loss` summary can read 0 if the final logged step
landed on an empty/last micro-batch — use `epoch/avg_loss` for the real per-epoch loss.)

### (d) attached files (per run) — good for reproducibility
| File | What |
|------|------|
| `code/scripts/ranking_loss_ref_fix.py` | **full snapshot of the trainer code** (~135 KB) as it ran |
| `wandb-metadata.json` | run provenance: **full CLI `args`** (incl. fsx/vlo/cft), **git remote + commit** (e.g. `ac9b95a9`), host (`slurm-node-*`), `gpu` (A40), `gpu_count`, `cudaVersion`, `python`, `os`, `slurm`, `startedAt` |
| `requirements.txt` | pip freeze of the run's environment |
| `config.yaml` | the config in (a) |
| `output.log` | captured stdout of the run |
| `wandb-summary.json` | the summary in (c) |

So a finished run is quite reproducible: exact code snapshot + git commit + full args + env
(`requirements.txt`) + delta-bins stats. The main missing piece is eval scores (separate).

### Caveat — this describes the v7/`ranking_loss_ref_fix.py` runs
The cloud project also has older runs (`ranking_loss_ref.py`, etc.) whose logged keys may differ
slightly; the schema above is for the current trainer (what all the paper-era runs use).
