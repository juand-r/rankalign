# Overnight run summary (kickoff snapshot — will be overwritten by job 37292)

Kicked off **Sun May 10 22:30 (UTC-5)** by Cursor agent.

This file will be **overwritten** with the actual morning report by Slurm
job **37292** once every dependency completes. If you're reading this in
the morning and the kickoff text below is still here, the morning-report
job hasn't run yet — check `squeue -u jdr` to see what's still pending.

See `overnight/jobids.txt` for the full job map.

## Quick-check commands

```bash
# Are the trainings still going?
squeue -u jdr -o "%.10i %.30j %.8T %.10M %R" | head -40

# Recent saves from any 2b ambigqa training job:
grep 'Saving to ' ~/logs/3727*.out 2>/dev/null | tail -10

# Did any job fail?
sacct --starttime now-12hours -u jdr --format=JobID,JobName%40,State,ExitCode,Elapsed | grep -E 'FAIL|TIMEOUT|CANCELLED'

# When the morning report finishes, this file will be replaced.
```

## Pipeline overview

```
   rosch 2b-it evals (already running)
      37268  37269 ────────────────┐
                                   │
   ambigqa 2b training x9         │
      37270..37278 ─────► ambigqa 2b matched eval (37288)──────────────┐
                       └► ambigqa 2b basetyp_extra eval (37289)────────┤
                                                                       │
   ambigqa 2b-it training x9                                           │
      37279..37287 ─────► ambigqa 2b-it matched eval (37290)───────────┤
                       └► ambigqa 2b-it basetyp_extra eval (37291)─────┤
                                                                       ▼
                                                       morning report (37292)
                                                       writes OVERNIGHT_STATUS.md
```

All eval-stage dependencies use `--dependency=afterany`, so an individual
training failure won't block the rest of the pipeline. The eval scripts
themselves skip missing model dirs.
