# Overnight state — 2026-05-25 ~02:15 CT (07:15 UTC)

User went to bed; full autonomy granted. This doc captures the state so the
morning agent (or user) can pick up quickly.

## Active autonomous worker

`scripts/_overnight_worker.sh` is running as a background daemon (PID was
`1009170`; check with `ps -ef | grep _overnight_worker`). It:

- Ticks every **20 min**
- Each tick:
  - Logs queue status
  - Runs `python scripts/_overnight_master.py` (Pass 1 eval gap-fills + Pass 2 train gap-fills, capped by `MAX_P1=6` per tick)
  - Every 3rd tick (~hourly): regenerates rosch + persona tables for all 5 metrics × {2b, 2b-it, 9b-it}
- Logs land in `docs/overnight_worker_logs/`
- Summary index: `docs/overnight_worker_logs/_summary.txt`

The agent wake loop (terminal `141348`) ticks every 60 min independently and
notifies the agent to inspect.

## Disk routing fix (committed `4b42a234`)

`/datastor1` is at 100% (~405 GB free of 153 TB; flatlining). All NEW eval
score CSVs from `_overnight_launch.sh` and `_eval_only.sh` now write to
`/datastor2/jdr/rankalign/outputs/` via a new `--outputs-dir` flag in
`build_eval_flags()`. Both v7 table builders already include `/datastor2`
in their `SEARCH_DIRS` so picked-up automatically.

Verified live: 42321 (persona×2b×s11 self NO_BASE) and 42330 (mem×9b-it×s4
self NO_BASE) wrote CSVs to `/datastor2/.../outputs/` at 02:09–02:10 CT.

## ifeval OOM fix (committed `9c288925`)

Single-GPU 2b/2b-it on `ifeval-concat` OOM-crashed at ~21min during the
consistency-ft pre-pass. Added `--gradient-checkpointing` to non-gemma-4
ifeval trains in `_overnight_launch.sh`. Re-submitted:

- `42331` ifeval × gemma-2-2b-it × s13 (RUNNING)
- `42334` ifeval × gemma-2-2b × s13 (RUNNING)
- `42332/42333`, `42335/42336` chained NO_BASE evals (PD)

## In-flight s13 (consistency-FT) trains

| jobid | dataset | model | status |
|-------|---------|-------|--------|
| 42114 | humaneval | 31B | RUNNING ~7h elapsed (24h limit) |
| 42256 | mem | 2b | RUNNING ~3h |
| 42259 | mem | 2b-it | **COMPLETED** (eval CSVs landed) |
| 42262 | mem | 9b-it | **COMPLETED** (eval CSVs landing) |
| 42265 | persona | 2b-it | RUNNING ~3h |
| 42283 | persona | 2b | RUNNING ~2h |
| 42286 | persona | 9b-it | RUNNING ~2h |
| 42310 | ifeval | 9b-it | RUNNING ~15min |
| 42331 | ifeval | 2b-it | RUNNING (retry) |
| 42334 | ifeval | 2b | RUNNING (retry) |

Plus `42307` persona × 9b-it × s1 (RUNNING). The two original ifeval
2b-it/2b s13 trains (42313/42316) FAILED (OOM, fixed above).

## Other priority gaps in `GAP_LIST` (Pass 2 will pick up as queue drains)

Top priorities (from `scripts/_overnight_master.py`):
- persona × 9b-it × s3 (skip — fully eval'd)
- ifeval × {2b-it, 2b} × {s1..s7, s11, s12} — many cells
- mem × {2b, 2b-it, 9b-it} × {s1..s7, s11, s12} — many cells
- persona × {2b, 2b-it} × {s1..s7, s11, s12} — many cells

Pass 2 also schedules NO_BASE chained evals as deps for s1/s2/s3/s13 trains.

## Where to look in the morning

- `docs/overnight_worker_logs/_summary.txt` — index of worker ticks
- `docs/overnight_worker_logs/<latest>_iter*.log` — full output of each tick
- `docs/overnight_worker_logs/{rosch,persona}_*_<stamp>.txt` — table snapshots from the worker
- `~/logs/<jobid>.{out,err}` and `/datastor2/jdr/logs/<jobid>.{out,err}` — slurm
- `squeue -u jdr` — current queue
- `sacct -u jdr -S 2026-05-25 --format=JobID,JobName%30,State,Elapsed -X` — full history
- `/datastor2/jdr/rankalign/outputs/` — fresh score CSVs

## To regenerate the tables yourself

```bash
cd /datastor1/jdr/gv-gap/rankalign
source /u/jdr/venvs/venv_lexcons/bin/activate
bash /tmp/_build_all_metrics.sh   # if you saved the script
# or hand-roll:
ROSCH_METRIC=gen_roc ROSCH_MODEL=9b-it python scripts/_build_rosch_table_v7.py
PERSONA_METRIC=gen_roc PERSONA_BASE=9b-it python scripts/_build_persona_v1_table_v7.py
PERSONA_METRIC=gen_roc python scripts/_persona_id_ood_split.py
```

## To stop the worker

```bash
pkill -f _overnight_worker.sh
```

## Constraints respected tonight

- All model dirs go to `/datastor2/jdr/rankalign/models2/` (per `save-models-to-datastor2.mdc`)
- All slurm logs go to `/datastor2/jdr/logs/` (per `~/.local/bin/run`)
- All NEW eval CSVs go to `/datastor2/jdr/rankalign/outputs/` (this session)
- Slurm queue cap respected (32 jobs max across PD+R)
- ifeval uses `--disc-shots zero` (avoids `NotImplementedError` in `make_prompt_ifeval`)
