# RankAlign inventory — take-stock (June 1–2, 2026)

Post-paper-submission stock-taking. Goal: know **where every trained model and eval
score lives**, what flags produced it, what training logs exist, and which paper results
were not epoch2. Initially built from `/datastor2/jdr/rankalign` + HuggingFace + repo docs
while **`/datastor1` was down** (NFS `10.202.8.210`, 2026-06-01).
**datastor1 came back 2026-06-02 and the second pass is DONE** — findings folded in below
(short version: v7/paper models were already complete on `/datastor2`+HF; datastor1 only
added the full **v6** archive). Training logs live *with* the models: gemma-2 as JSON in
`models2/training_run_logs/`, qwen as `training_log.log.gz` inside each HF repo; only gemma-4 cu
s3/s4/s7 have no preserved log (see `WANDB_AND_LOGS.md` §2). No pod revival needed.

> Rule for this folder: **additive only.** Nothing here deletes or moves data; it documents.

## Structure

```
inventory-jun1-2026/
├── README.md                       ← this file
├── SETTINGS_REFERENCE.md           ← what s1..s13 mean (verified from source) — READ FIRST
├── v7/
│   └── training_inventory_v7.md    ← TASK 1, v7 + v7b (the paper models)
├── v6/
│   └── training_inventory_v6.md    ← TASK 1, v6 (older, for completeness)
├── TASK2_eval_inventory.md         ← TASK 2 (scores_ files + metrics CSVs)   [in progress]
├── TASK3_non_epoch2_in_paper.md    ← TASK 3 (paper results that aren't epoch2) [in progress]
├── _build_training_inventory.py    ← regenerates v7/ + v6/ tables from _raw/
├── _dump_hf_repos.py               ← refreshes _raw/hf_repos.json
└── _raw/                           ← captured source data (audit trail, reproducible)
    ├── models2_listing.txt         271 v7 local adapter dirs (/datastor2/.../models2)
    ├── datastor1_models_listing.txt 1319 v6 dirs — the FULL v6 archive (/datastor1/.../models); v6 table built from this
    ├── models_v6_listing.txt       201 v6 dirs — partial /datastor2/.../models copy (superseded by datastor1)
    ├── hf_repos.json               latkes (17) + TAUR-dev (326) rankalign model repos
    ├── hf_checkpoint_map.json      authoritative 40-entry local→HF map (partial)
    ├── auto_delta_log.csv          delta-bins auto-computed delta per run
    ├── training_run_logs_listing.txt  85 per-run training JSON logs
    └── parsed_all.csv              every model row parsed (813 rows, 0 unparsed)
```

## How to read the training inventory tables

Each row = one (model, task, setting). The `e0 / e1 / e2` columns show **where that epoch's
checkpoint lives**:
- `L` = local on-disk: `/datastor2/jdr/rankalign/models2` (v7) or `/datastor1/jdr/gv-gap/rankalign/models` (v6 — the full archive)
- `latkes` = HuggingFace `latkes/…`
- `TAUR` = HuggingFace `TAUR-dev/…`
- `·` = checkpoint not found anywhere we can currently see

Setting meanings (s1..s13) are in `SETTINGS_REFERENCE.md`.

## Known caveats baked into the data

- **Delta sweeps:** persona-v1 `s11`/`s12` rows list many delta values
  (0.047…, 0.095…, 0.158…). Those are the **delta-bins sweep** experiments, not one
  canonical run — each delta is a separate small run, mostly e0/e1 only.
- **`comb-notc-nofsx?`** rows: a combined-loss run with neither TC nor force-same-x — not
  one of the canonical numbered settings (likely a sweep artifact). Flagged with `?` so it
  isn't mistaken for a paper setting.
- **`d2g--random--alpha1.0`** is common to every dir (generator-direction, random split,
  alpha 1.0) — omitted from the per-row notes since it never varies.

## Training logs (TASK 1c) — where they are

- **Per-run JSON:** `/datastor2/jdr/rankalign/models2/training_run_logs/*.json`
  (85 files, naming `<timestamp>_google--<model>_<task>.json`). **The setting is NOT in the
  filename** — it must be read from the JSON's recorded args (delta config, loss weights,
  fsx, cft `consistency_ft` block, label partition, seed). **Setting→file map done:**
  `v7/training_logs_map_v7.md`. Full JSON schema + examples: `WANDB_AND_LOGS.md`.
- **v6 logs:** older runs may not have the JSON (the per-run JSON convention started ~May 24).
- **wandb:** authoritative source is the **cloud** `juand-r/rankalign` (1339 runs); see
  `WANDB_AND_LOGS.md` + `WANDB_RUN_MAP.md`. mll runs synced online; **gemma-4 cu/cm pods ran
  `WANDB_MODE=offline` and did NOT sync → no usable gemma-4 curves** (the local
  `/datastor2/.../wandb/` dir has only persona, May 19–23).
- **gemma-4 (cu) provenance:** see `../provenance-cu-s2-rankalign/` for the RankAlign (s2)
  run's full training log, config JSON, and HF-upload log.

## Status (2026-06-02)

- [x] SETTINGS_REFERENCE.md — verified from source scripts
- [x] TASK 1 v7/v6 tables — model locations + epochs + delta (auto-generated)
- [x] TASK 2 — eval scores + metrics CSV inventory (`TASK2_eval_inventory.md` + `v7/eval_inventory_v7.md`)
- [x] TASK 3 — non-epoch2 paper results (`TASK3_non_epoch2_in_paper.md`)
- [x] TASK 1c — training JSON log → setting map (`v7/training_logs_map_v7.md`, 85 logs classified from flags)
- [x] WandB coverage + JSON-log structure (`WANDB_AND_LOGS.md`) — cloud `juand-r/rankalign` (1339 runs) is authoritative: **ifeval + rosch/membership have wandb** (all gemma-2, online); **humaneval-cu (gemma-4) is the one gap** (offline-on-pod, not synced). Plus the full JSON-log schema.
- [x] WandB run→model precision map (`WANDB_RUN_MAP.md`) — every paper model×setting → its wandb run name+URL+state (82 finished / 64 failed / 64 crashed of 210 paper-task runs)
- [x] **datastor1 second pass (DONE 2026-06-02)** — `/datastor1/.../models` = full v6 archive
  (1319 dirs, all v6; 0 v7/gemma-4/qwen). v6 table rebuilt from it; v7 was already complete on
  /datastor2+HF. datastor1 also holds the canonical `outputs/` (mirror of the scores already
  inventoried) + 32 wandb dirs (cloud is authoritative). No v7 gap on datastor1.
- [x] enumerate `/datastor2/.../outputs/` for the gemma-2-2b/2b-it scores (TASK2 follow-up) — 20,402 found
- [x] training-log locations resolved: gemma-2 → JSON in `models2/training_run_logs/`; qwen → `training_log.log.gz` in each HF repo; gemma-4 cu s2/s13 → JSON on disk. Only **gemma-4 cu s3/s4/s7** have no preserved log (HF adapter-only) — known gap, no pod revival.

### Headline findings
- **Non-epoch2 in the paper:** gemma-4 correct-upper **RankAlign=epoch1** (e2 exists, eval-only fix)
  and **SFT=v6/epoch1** (no v7 model); **s13 (SFT+cft) = epoch0 everywhere** (gemma-4 cu, qwen
  ifeval+membership) — it ran out of time. Everything else in the final tables is epoch2.
- **Models:** v7 adapters live on `/datastor2/.../models2` (271) + HF latkes (17) + TAUR-dev (326);
  gemma-4 cu adapters are on **latkes** (s2 d2.14 e0/e1/e2 private; s13 e0 public).
- **One eval produced NO score files** (rosch OSError-36) — metrics salvaged from stdout into
  `docs/salvaged_rosch_metrics.md`.
