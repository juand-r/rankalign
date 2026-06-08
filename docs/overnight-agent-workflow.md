# Overnight Job Monitoring Workflow (Cursor Agent)

How the Cursor IDE agent (not Claude Code CLI) sets up unattended overnight
job submission, monitoring, and error recovery on the Slurm cluster.

## Overview

The agent can stay active after the user goes to sleep by running a background
shell with a monitoring script. The script polls `squeue`, launches follow-up
jobs when predecessors finish, diagnoses failures, and logs everything.

## Architecture

```
User goes to sleep
        │
        ▼
┌─────────────────────────────────────────────┐
│  Background Shell (block_until_ms=0)        │
│  └── _overnight_monitor.sh                  │
│       ├── Phase A: poll squeue until done   │
│       ├── Phase B: launch next batch        │
│       ├── Phase C: poll squeue until done   │
│       └── Phase D: report + exit            │
└─────────────────────────────────────────────┘
        │
        ▼  (notify_on_output regex)
┌─────────────────────────────────────────────┐
│  Cursor Agent receives notifications        │
│  on pattern matches (phase transitions,     │
│  failures, completion).                     │
│  Can intervene if needed.                   │
└─────────────────────────────────────────────┘
```

## Step-by-Step Setup

### 1. Prepare launch scripts

Write the batch launcher scripts (e.g. `run_trainset_dynamics_ifeval_all_9bit.sh`)
and verify with `DRYRUN=1` that all model paths resolve and job counts are correct.

### 2. Write the monitor script

The monitor script is a simple bash loop with phases. Key pattern:

```bash
#!/bin/bash
set -euo pipefail
LOG="/path/to/overnight/_monitor.log"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOG"; }

# Phase A: wait for running jobs
while true; do
    N=$(squeue -u jdr --noheader --format="%j" | grep -c "^myjob-" || true)
    N=${N:-0}
    [ "$N" -eq 0 ] && break
    log "Still running: $N. Sleeping 5 min..."
    sleep 300
done
log "PHASE A DONE"

# Phase B: launch next batch
bash scripts/my_launcher.sh 2>&1 | tee -a "$LOG"
log "PHASE B DONE"

# Phase C: monitor new jobs
# ... same pattern as Phase A ...

# Phase D: check outcomes with sacct, count output files
```

### 3. Launch as background shell with notifications

```
Shell tool call:
  command: "bash scripts/_overnight_monitor.sh"
  block_until_ms: 0
  notify_on_output:
    pattern: "PHASE .* DONE|FATAL|WARNING|ALL.*SUCCEEDED|ACTION NEEDED"
    reason: "overnight phase transitions"
    debounce_ms: 60000
```

This immediately backgrounds the script. The agent gets notified on phase
transitions and can respond (or just acknowledge).

### 4. Smoke check

After launching, do a quick `head -20` on the terminal output file to confirm
the script started correctly and isn't erroring on line 1.

### 5. Handle notifications

When notified:
- **PHASE X DONE**: Acknowledge, no action needed.
- **WARNING/FATAL**: Read the log, diagnose, potentially kill and relaunch.
- **ALL JOBS SUCCEEDED**: Done. Report to user when they return.

## Failure Recovery Pattern (Example: Wrong VENV)

1. Monitor reports "ALL JOBS SUCCEEDED" but score file count = 0.
2. Agent checks job stderr: finds `ValueError: model_type qwen3_5 not recognized`.
3. Agent identifies fix: set `VENV=/datastor2/jdr/venvs/qwen35`.
4. Agent updates the launcher script, relaunches, starts a new monitor.
5. New monitor confirms jobs actually produce output this time.

## Key Lessons Learned

- **Always verify job outputs, not just exit codes.** A job can exit 0 but
  produce nothing if the eval script catches exceptions internally.
- **Do a 10-minute sanity check** after launching a new batch: read the first
  job's stderr to confirm the model loads and inference starts.
- **Use `sacct` for final state**, not `squeue` (which only shows running jobs).
- **Poll interval of 5 minutes** is a good balance between responsiveness and
  not hammering the scheduler.
- **Log everything to a file** (`tee -a`) so the user can review the full
  timeline in the morning.
- **Include diagnostics in the monitor**: check for OOM, model-type errors,
  timeouts. Don't just report "failed" — say why.

## Regex Tips for notify_on_output

The agent's `notify_on_output` uses regex matched against stdout/stderr of
the background shell. Good patterns:

- Phase transitions: `"PHASE [A-D] DONE"`
- Errors: `"FATAL|ACTION NEEDED"`
- Completion: `"ALL.*SUCCEEDED|MONITORING COMPLETE"`
- Combine: `"PHASE .* DONE|FATAL|WARNING|SUCCEEDED|ACTION NEEDED"`

Set `debounce_ms: 60000` (1 min) to avoid notification spam during rapid
log output (e.g. during batch submission).

## File Locations

- Monitor scripts: `scripts/_overnight_monitor*.sh` (ephemeral, not always committed)
- Logs: `overnight/_overnight_monitor*.log`
- Job ID tracking: `overnight/_trainset_dynamics_jobids.txt`
- Score outputs: `/datastor2/jdr/rankalign/outputs-trainset-dynamics/`
