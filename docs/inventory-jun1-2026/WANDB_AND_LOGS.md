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

### What actually exists on disk (103 local wandb run dirs, ALL dated May 19–23)
All under `/datastor2/jdr/rankalign/scripts/wandb/` (+ 1 in `…/rankalign/wandb/`). By cell:

| Task | Models with wandb | Settings | Notes |
|------|-------------------|----------|-------|
| **persona-v1** | gemma-2-2b, gemma-2-2b-it, gemma-2-9b-it | **~all (s1–s12)** incl. delta-bins sweeps | ~95 runs; mix of fixed-`delta0.15` and delta-bins. **This is essentially all the wandb we have.** |
| ifeval | qwen3.5-9b | s2 | 4 stray early runs |
| membership | qwen3.5-9b | s2 | 1 stray early run |
| 3sat | Qwen2.5-7B-Instruct | s11 | 1 stray (a different side experiment) |

### Bottom line — who has wandb and who doesn't
- ✅ **HAS wandb:** **persona-v1** for all three gemma-2 models (basically every setting), online in
  the `rankalign` project + local dirs on mll. Plus a few stray early qwen3.5-9b s2 runs.
- ❌ **NO wandb:** **membership (rosch)**, **ifeval** (the gemma-2/qwen paper cells), and
  **humaneval-cu / humaneval-cm** (gemma-4). Also **every v7b run** and **the entire May 24–25
  batch** (`--no-wandb`).
- ⚠ **gemma-4 cu/cm:** wandb was *enabled but offline*, so run dirs sat on the pods that are now
  stopped — not synced to the cloud, not copied to mll. Treat as lost unless a pod volume is revived.

### Important caveat (don't over-trust persona wandb)
The 103 wandb runs are **May 19–23** — the *earlier* persona batch (mostly fixed-`delta0.15`
+ sweeps). The **canonical May 24–25 persona runs** (the ones with JSON logs, delta-bins) have
**no wandb**. So even for persona, the final paper models likely rely on the JSON logs below, not
wandb. wandb is best seen as exploratory-era curves for persona, not a per-final-model record.

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
