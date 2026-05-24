# TODO: Launch s13 (SFT + `--consistency-ft`) train + eval

Created 2026-05-24. **Not launched yet** — cluster is saturated by the
overnight s1–s12 backlog. When that drains, run these.

For full design + flag rationale see
[`docs/s13_consistency_ft.md`](s13_consistency_ft.md).
For the running changelog entry see
[`docs/UPDATES.md`](UPDATES.md) (2026-05-24).

## Prerequisites (verify before launching)

```bash
cd /datastor1/jdr/gv-gap/rankalign
squeue -u $USER -o "%i %T %D %C %m %b %M %l %j" | head    # queue light?
df -h /datastor2                                          # space?
git log --oneline -1 -- scripts/ranking_loss_ref_fix.py    # has cft commit?
```

Smoke any single cell first:

```bash
DRYRUN=1 bash scripts/_overnight_launch.sh humaneval gemma-4-31B-it s13
```

## The 10 cells to launch

Each call submits **1 train job + 2 eval jobs** (TC variants `self` and
`neg`, both chained `--dependency=afterany:<train_jobid>`). 10 cells = 30
slurm jobs at full saturation. Walltime per cell is set conservatively in
the dispatcher; can be overridden via `WALLTIME=H bash ...`.

```bash
# 3 × membership-sans-rosch-v0  → eval on 10 rosch-* tasks
bash scripts/_overnight_launch.sh membership gemma-2-2b      s13   # 1 GPU,  12 h
bash scripts/_overnight_launch.sh membership gemma-2-2b-it   s13   # 1 GPU,  12 h
bash scripts/_overnight_launch.sh membership gemma-2-9b-it   s13   # 1 GPU,  24 h

# 3 × persona-v1                → eval on 6 persona-v1 test tasks
bash scripts/_overnight_launch.sh persona    gemma-2-2b      s13   # 1 GPU,  10 h
bash scripts/_overnight_launch.sh persona    gemma-2-2b-it   s13   # 1 GPU,  10 h
bash scripts/_overnight_launch.sh persona    gemma-2-9b-it   s13   # 1 GPU,  20 h

# 3 × ifeval-concat             → eval on 21 ifeval-prompt_* tasks
bash scripts/_overnight_launch.sh ifeval     gemma-2-2b      s13   # 1 GPU,  14 h
bash scripts/_overnight_launch.sh ifeval     gemma-2-2b-it   s13   # 1 GPU,  14 h
bash scripts/_overnight_launch.sh ifeval     gemma-2-9b-it   s13   # 2 GPUs, 30 h

# 1 × humaneval-v2.1correct-upper → eval on 82 humaneval-* tasks
bash scripts/_overnight_launch.sh humaneval  gemma-4-31B-it  s13   # 3 GPUs, 24 h
```

## Special-case reminders

- **gemma-4-31B-it** auto-triggers (in `_overnight_launch.sh`):
  - venv = `/datastor2/jdr/venvs/gemma4` (transformers 5.x)
  - `--gemma4-lora` (no `_merged` sibling produced)
  - `--gradient-checkpointing`
  - `--disc-shots zero` (per `run_settings_v21correct_upper.sh`)
  - 3 GPUs / 192 GB / 24 h
- **gemma-2-9b-it × ifeval** auto-allocates 2 GPUs (model-parallel) and a
  longer 30 h walltime — Slurm's default `bf_window` is 24 h, so this cell
  may need backfill on a less-contended scheduler. Consider `WALLTIME=20`
  if the queue is hot.
- **`/datastor2` only** for everything: `MODELS_DIR` defaults to
  `/datastor2/jdr/rankalign/models2` and slurm logs go to `/datastor2/jdr/logs/%j.{out,err}`.
- All 10 calls write to `overnight/_overnight_jobids.txt` so the existing
  monitor loop / progress notes can pick them up.

## Post-launch sanity checks

After the first train job starts (~5-15 min), confirm:

```bash
# 1) cft argparse didn't reject anything
grep -E "consistency-ft|CONSISTENCY-FT" /datastor2/jdr/logs/<JOBID>.out

# 2) The pre-pass actually ran (look for the banner)
grep -E "CONSISTENCY-FT FILTER|t_v.*t_g|kept.*labeled-agree" /datastor2/jdr/logs/<JOBID>.out

# 3) JSON log captured the stats
ls -lt /datastor2/jdr/rankalign/models2/training_run_logs/ | head
jq '.consistency_ft' /datastor2/jdr/rankalign/models2/training_run_logs/<latest>.json
```

After eval finishes, score CSVs land under `outputs/scores_*--cft--*.csv`
(the `--cft` substring distinguishes them from s1 / SFT-lo runs).

## Recovery

If a train job dies mid-run, the dispatcher's eval-side `ls -dt … | head -1`
will pick the most-recent saved epoch (typically `epoch1` or `epoch2`),
so the eval still runs. If a train fails BEFORE the first epoch save, the
eval will print `SKIP - no matching ...` and exit 0 — re-launch via
`bash scripts/_overnight_launch.sh <DATASET> <MODEL> s13` (the dispatcher
re-submits a fresh train; no idempotence layer here).

## Done criteria

s13 is "done" when:

- All 10 train cells have at least an `epoch1` or `epoch2` adapter on disk.
- Per-cell scores CSVs exist for both TC variants (`self` and `neg`)
  across all eval tasks.
- A `_build_*_table_v7.py`-style table-builder run shows s13 alongside
  s1–s12 with no missing cells. (May need a small extension to the
  v7 table builders to recognize the `--cft` substring; check before
  building.)
