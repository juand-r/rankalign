# Phase A — backfill no-base TC evals (PMI-self + Neg-self columns)

Started 2026-05-25 ~02:03 UTC, in response to: *"What I want: the full
tables with s1 - s12 done, with all required evals done (base typ and
without base typ). For those which are not trained with TC, I want both
self and Neg TC evals too."*

## What this fills

The v7 dispatcher (`scripts/_overnight_launch.sh`) only ever runs
`--self-typcorr --base-typcorr` or `--neg-typcorr --base-typcorr` evals.
That fills the `basetyp-` / `basetypneg-` CSV families → "PMI base" /
"Neg base" columns of the rosch / persona-v1 tables.

The "PMI self" and "Neg self" columns (CSV prefixes `self-` / `neg-`)
require a different eval invocation: `--self-typcorr` or `--neg-typcorr`
**without** `--base-typcorr`. As of 2026-05-25 there were
**zero** v7 `self-*` or `neg-*` CSVs on disk. Phase A submits these.

## Mechanism

`scripts/_eval_only.sh` was extended on 2026-05-24 to accept `NO_BASE=1`:

```bash
# basetyp-* (already-done by dispatcher)
bash scripts/_eval_only.sh DATASET MODEL SETTING TC

# self-* / neg-* (Phase A — adds --log-odds, drops --base-typcorr)
NO_BASE=1 bash scripts/_eval_only.sh DATASET MODEL SETTING TC
```

Per-cell coverage rule:

| Setting          | TC variants needed (with NO_BASE=1) |
|------------------|-------------------------------------|
| s1, s2, s3       | `self` AND `neg` (no-TC trained)    |
| s4, s5, s6, s11  | `self` (self-TC trained)            |
| s7, s12          | `neg` (neg-TC trained)              |
| s13              | both `self` and `neg`               |

## Total Phase A scope

48 jobs across the four (model, dataset) blocks:

- membership × 2b-it: 12 (all done, no afterany)
- membership × 9b-it: 12 (s2-s12 done, s1=42021 needs afterany)
- persona × 2b-it: 12 (all done, no afterany)
- persona × 9b-it: 12 (s2/s5/s6 done; s1/s3/s4/s7/s11/s12 need afterany on respective trains)

**Out of scope for Phase A** (no models on disk yet):
- ifeval × {2b-it, 9b-it}
- humaneval × g4-31B-it (s13 only, train=42114 still running)

## Throttling

User cap is 32 (PD+R) jobs total. We submit batches of ~12 each tick
(60-min wake loop) until all 48 are landed.

## Batches submitted

### Batch 1 — 12 jobs, 2026-05-25 02:03 UTC

| Job   | Dataset | Model      | Setting | TC   | Notes |
|-------|---------|------------|---------|------|-------|
| 42140 | mem     | 9b-it      | s2      | self | immediate |
| 42141 | mem     | 9b-it      | s2      | neg  | immediate |
| 42142 | mem     | 9b-it      | s3      | self | immediate |
| 42143 | mem     | 9b-it      | s3      | neg  | immediate |
| 42144 | mem     | 9b-it      | s4      | self | immediate |
| 42145 | mem     | 9b-it      | s7      | neg  | immediate |
| 42146 | mem     | 9b-it      | s11     | self | immediate |
| 42147 | mem     | 9b-it      | s12     | neg  | immediate |
| 42148 | persona | 9b-it      | s2      | self | immediate |
| 42149 | persona | 9b-it      | s2      | neg  | immediate |
| 42150 | mem     | 9b-it      | s5      | self | immediate (after vallogodds fix) |
| 42151 | mem     | 9b-it      | s6      | self | immediate (after vallogodds fix) |

Mid-batch fix: `_eval_only.sh` had `VLO_STR=--vallogodds` for s5/s6, but
`_overnight_launch.sh` drops `--validator-log-odds` for these settings
(per IRP §1). Patched and the two skipped jobs (mem/9b-it/s5/s6 self)
were resubmitted as 42150/42151. Committed: `562aa108` → rebased onto
`a4c534b7`.

### Batches still to submit

- **Batch 2** (when queue < 20 jobs): mem × 2b-it × all settings (~12)
- **Batch 3** (when queue < 20): persona × 2b-it × all settings (~12)
- **Batch 4** (when queue < 20): persona × 9b-it remaining + mem × 9b-it × s1
  (afterany on the running trains). 12 jobs.

## Recovery rules

- If an `_eval_only.sh` invocation prints `FATAL: no model dir matching
  glob`, the setting fingerprint in `_eval_only.sh` doesn't match what
  the launcher actually produced. Inspect the actual model dir under
  `/datastor2/jdr/rankalign/models2/v7-google--<model>--*` to find the
  missing/extra flag token, patch the relevant `s#)` block, retry.
- If an eval crashes with `OSError: [Errno 36] File name too long`,
  the eval script's filename hashing path is broken — check
  `src/tasks/common.py:build_csv_filename` and the abbreviation map in
  `src/checkpoint_name_parser.py`.
- If a training job fails before `epoch1` save, the matching
  `--dependency=afterany` evals will print `SKIP - no matching ...` and
  exit 0; relaunch via `bash scripts/_overnight_launch.sh ...`.

## After Phase A completes

1. Re-run `_status_matrix.py` → verify all `done` cells now have
   matching `self-` or `neg-` CSVs.
2. Extend `_build_rosch_table_v7.py` and `_build_persona_v1_table_v7.py`
   to look at no-base CSV prefixes for the "PMI self" / "Neg self"
   columns. (The current builders may already pick these up — verify
   with a dry run.)
3. Refresh `docs/v7_tables_snapshot_2026-05-24.md`.
