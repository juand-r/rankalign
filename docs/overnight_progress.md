# Overnight Progress Log (2026-05-24)

Live log of bash commands, sbatch IDs, status, and decisions made
overnight. Most recent entries on top. Companion to
[`overnight_run_instructions.md`](overnight_run_instructions.md) and
[`overnight_plan.md`](overnight_plan.md).

## Quick state cheat-sheet

- Loop interval: 30 min (sentinel: `AGENT_LOOP_TICK_overnight`).
- Slurm cap: 20 RUNNING, ≤32 total in queue at any time.
- Models go to `/datastor2/jdr/rankalign/models2/`. Logs to
  `/datastor2/jdr/logs/`.

## Commands and events (newest first)

### 10:50 — Loop tick #15: slurm predicted 10:26 START but didn't materialize

41855's `StartTime` field still says 15:26 UTC but it's now 15:50 UTC
and the job is still PENDING. Slurm prediction was a best-estimate,
not a guarantee. node-001 and node-002 each have 2 GPUs free but
slurm reserves them for higher-priority pending jobs (backfill
protection). chizhang's array still has ~2h to go. Nothing else
freeing up imminently.

Action: continuing to hold. No forward progress in 7h. If still
nothing by tick #20 (~13:20), I'll cancel a few mid-priority jobs to
free fair-share weight on the high-priority ones.

### 07:20 — Loop tick #8: still 0 R, 108 PD; predictions stable

41855 still predicted to start at 10:26 CT (~3h). No state changes in
the last hour. Action: none.

### 06:50 — Loop tick #7: nothing changed; predictions stable

- 0 R, 108 PD. Predicted start times unchanged.
- Next big GPU release: pier 41712_48 ends ~9:30 CT (2.6h);
  chizhang array ends ~13:00 CT (6h). Consistent with slurm's
  predicted 10:26 CT start for my 41855.
- Action: none, wait.

### 06:20 — Loop tick #6: slurm StartTime predictions are LATE

`scontrol show job` reveals slurm's planned start times for my first
queued jobs:
- 41855 (persona × 2b-it × s4): 10:26 CT today (4h from now)
- 41857 (persona × 2b-it × s7): 17:08 CT today (11h)
- 41859 (persona × 2b-it × s2): 20:27 CT today (14h)
- 41861-41869: ~9:42 CT MONDAY morning (22h)

User probably wakes ~7-9am CT, so only a handful of trains will have
*started* by then. Cause: chizhang's job array (2-day walltime, ~7h
left) and jocelyn's 15 jobs (2-day walltime, 38h+ left) own most GPU
slots. Backfill can't easily fit my 10-30h jobs into the remaining
small windows.

Decision: do NOT cancel+resubmit. Cancelling resets age-priority and
would push StartTimes EVEN later. Trust slurm's plan; wait it out.

### 05:50 — Loop tick #5: cluster contended; still 0 R, 108 PD

- All 108 jobs still PD with reason "Priority". My priority value:
  `12026` (uniform across my batch).
- Cluster: 6 nodes in `mixed-` (draining, won't accept new jobs);
  5 nodes in `mixed` saturated by other users (chizhang at 8 GPUs/job
  on multiple nodes, jocelyn at 12 GPUs/job, qdf76 at 8 GPU/job).
  Total cluster jobs: 34.
- Action: nothing to do. Wait for other users' jobs to finish.
  My longer walltimes hurt backfill chances slightly but they're
  necessary so no point shortening.

### 05:25 — Loop tick #4: walltimes were too short. Cancelled+resubmitted.

Discovered 41747 was at ~2s/it after 41m, ETA ~8.5h for 3 epochs but
walltime was only 4h. ALL queued jobs had similar undersized walltimes
=> they would all walltime-TIMEOUT before epoch2 saved, and their
eval-globs (hardcoded `epoch2`) would SKIP. Both bugs needed fixing.

Actions:
- Updated [`scripts/_overnight_launch.sh`](../scripts/_overnight_launch.sh):
  bumped walltimes 2-3x, loosened eval glob to `epoch[012]`.
- Cancelled all 107 PD jobs + cancelled the running 41747 (cost ~45m
  of progress; necessary to get a properly-sized walltime).
- Resubmitted Phase 1A + 1B + 2 with new launcher.
- Committed `d50cf2c4`/`fea681ef` to origin/longform.

New job IDs: 41855-41962 (108 jobs, all PD). Slurm picks up scheduling
again from priority queue.

### 04:50 — Loop tick #3: smoke PASSED, first job RUNNING

- 41730 TIMEOUT @ 5h walltime (expected, baseline ref preserved at epoch1).
- **41745 smoke COMPLETED in 3:55** — `--shape-budget-mode global` +
  per-item disc-gating perf fix validated end-to-end. Smoke saved to
  `/datastor2/jdr/rankalign/models2-smoke-bc-global/` (we can ignore /
  delete that later).
- **41747 RUNNING for 10m** (persona × 2b-it × s4: fsx+ppd+sbm-global).
  Loaded data (5110 samples), made dataloader, no errors. Per-run JSON
  log written. The riskiest new-flag combo is happily training.
- Cluster: many nodes in `mixed-` (drain-after-current) state; only
  5 nodes accepting new jobs. GPU bottleneck is the only thing
  slowing us. Slurm will start more as GPUs free.
- Queue: 108 (1 R, 107 PD).
- No failures.

### 04:20 — Loop tick #2: still waiting on 41730

- 41730 at 4:45:05 / 5:00:00; ~15m until walltime kill.
- Queue unchanged: 1 R, 109 PD. No failures.
- Action: none.

### 03:49 — Loop tick #1: nothing has started yet

- Queue: 1 R (41730 baseline at 4:14h / 5h walltime, ~45m left), 109 PD.
- No failed/cancelled jobs in the last hour.
- /datastor1 481G free; /datastor2 132T free. Disk fine.

Action: none. Wait for 41730 walltime kill at ~04:35 to free node-007.

### 03:34 — User said "go greedy". Phase 1B + Phase 2 submitted.

User went to bed; relaxed the 32-job cap with "submit many things and
hope things get scheduled". Submitted everything that was queued in
the plan:

```bash
bash scripts/_overnight_phase1b.sh   # 9b-it × 5 priority settings × 3 datasets
bash scripts/_overnight_phase2.sh    # s5,s6,s11,s12 × 2 models × 3 datasets
```

Phase 1B job IDs (train,eval pairs):
- 9b-it × persona × s4: 41777,41778
- 9b-it × membership × s4: 41779,41780
- 9b-it × ifeval × s4: 41781,41782
- 9b-it × persona × s7: 41783,41784
- 9b-it × membership × s7: 41785,41786
- 9b-it × ifeval × s7: 41787,41788
- 9b-it × persona × s2: 41789,41790
- 9b-it × membership × s2: 41791,41792
- 9b-it × ifeval × s2: 41793,41794
- 9b-it × persona × s3: 41795,41796
- 9b-it × membership × s3: 41797,41798
- 9b-it × ifeval × s3: 41799,41800
- 9b-it × persona × s1: 41801,41802
- 9b-it × membership × s1: 41803,41804
- 9b-it × ifeval × s1: 41805,41806

Phase 2 job IDs:
- 2b-it × {persona, membership, ifeval} × s5: 41807-41812
- 9b-it × {persona, membership, ifeval} × s5: 41813-41818
- 2b-it × {persona, membership, ifeval} × s6: 41819-41824
- 9b-it × {persona, membership, ifeval} × s6: 41825-41830
- 2b-it × {persona, membership, ifeval} × s11: 41831-41836
- 9b-it × {persona, membership, ifeval} × s11: 41837-41842
- 2b-it × {persona, membership, ifeval} × s12: 41843-41848
- 9b-it × {persona, membership, ifeval} × s12: 41849-41854

Total queue: 110 jobs (1 R, 109 PD). Slurm accepted all submissions.

### 03:30 — Phase 1A launched (gemma-2-2b-it × 5 priority settings × 3 datasets)

Submitted 15 train jobs (each with 1 chained eval = 15 evals on `afterany`):

```bash
cd /datastor1/jdr/gv-gap/rankalign
for setting in s4 s7 s2 s3 s1; do
  bash scripts/_overnight_launch.sh persona     gemma-2-2b-it $setting
  bash scripts/_overnight_launch.sh membership  gemma-2-2b-it $setting
  bash scripts/_overnight_launch.sh ifeval      gemma-2-2b-it $setting
done
```

Job IDs (train,eval pairs):
- persona × 2b-it × s4: 41747,41748 (eval tc=self)
- persona × 2b-it × s7: 41749,41750 (eval tc=neg)
- persona × 2b-it × s2: 41751,41752 (eval tc=self)
- persona × 2b-it × s3: 41753,41754 (eval tc=self)
- persona × 2b-it × s1: 41755,41756 (eval tc=self)
- membership × 2b-it × s4: 41757,41758 (eval tc=self)
- membership × 2b-it × s7: 41759,41760 (eval tc=neg)
- membership × 2b-it × s2: 41761,41762 (eval tc=self)
- membership × 2b-it × s3: 41763,41764 (eval tc=self)
- membership × 2b-it × s1: 41765,41766 (eval tc=self)
- ifeval × 2b-it × s4: 41767,41768 (eval tc=self)
- ifeval × 2b-it × s7: 41769,41770 (eval tc=neg)
- ifeval × 2b-it × s2: 41771,41772 (eval tc=self)
- ifeval × 2b-it × s3: 41773,41774 (eval tc=self)
- ifeval × 2b-it × s1: 41775,41776 (eval tc=self)

Queue is at cap (32 / 32). Phase 1B (9b-it) deferred until queue drops
below ~15 PENDING.

### 03:14 — Session start

User went to bed; gave overnight instructions. Saved verbatim to
`docs/overnight_run_instructions.md`. Started this progress log and
armed a 30-min stay-awake loop.

State on entry:
- 41745 (smoke `--shape-budget-mode global`): PENDING (Priority).
  When it runs it also exercises the per-item disc-forward gating
  perf fix.
- 41730 (baseline reference): RUNNING ~3h41m / 5h.
- No other agent jobs in queue.

### 11:52 — Cancel all ifeval-concat jobs (per user)

User asked to cancel all `ifeval-concat` jobs (both 2b-it and 9b-it). The
9b-it × ifeval set was already skipped this morning; this round removes
the 2b-it × ifeval set too.

Cancelled (9 trains + 9 dependent evals = 18 total):

```
scancel 41977 41987 41997 42007 42017 42027 42037 42047 42057 \
        41978 41988 41998 42008 42018 42028 42038 42048 42058
```

Train jobs (one per setting): 41977 (s2), 41987 (s7), 41997 (s2), 42007
(s3), 42017 (s1), 42027 (s5), 42037 (s6), 42047 (s11), 42057 (s12). The
overnight log records the (setting, model, dataset) mapping; mapping
jobid→cell is in `overnight/_overnight_jobids.txt`.

Queue after: 1 running (41973 = persona × 2b-it × s4), 71 pending = 36
train + 36 eval (persona-v1 and membership-sans-rosch-v0 only, 9
settings × 2 models = 18 cells × 2 jobs each).

### 11:54 — Cancel + resubmit all non-running 2b-it jobs at WALLTIME=5

User: "I also want you to cancel and resubmit all the still non-running
jobs that use 2b or 2b-it, and re-launch with less resources. I think
these would only need 5 hours to train."

Currently running: 41973 (persona × 2b-it × s4) — kept untouched. Its
paired eval 41974 (PD) was also kept since it depends on 41973.

Cancelled 17 trains + 17 evals (34 jobs):
```
scancel 41975 41983 41985 41993 41995 42003 42005 42013 42015 42023 \
        42025 42033 42035 42043 42045 42053 42055 \
        41976 41984 41986 41994 41996 42004 42006 42014 42016 42024 \
        42026 42034 42036 42044 42046 42054 42056
```

Resubmitted same 17 cells with WALLTIME=5 via:
```
for cell in "membership s4" "persona s7" "membership s7" "persona s2" "membership s2" \
            "persona s3" "membership s3" "persona s1" "membership s1" \
            "persona s5" "membership s5" "persona s6" "membership s6" \
            "persona s11" "membership s11" "persona s12" "membership s12"; do
  read -r DATASET SETTING <<< "$cell"
  WALLTIME=5 bash scripts/_overnight_launch.sh "$DATASET" "gemma-2-2b-it" "$SETTING"
done
```

New train jobids (PD, 5h walltime, 1 GPU, 64 GB):
- membership×s4: 42064 (eval 42065)
- persona×s7:    42066 (eval 42067)
- membership×s7: 42068 (eval 42069)
- persona×s2:    42070 (eval 42071)
- membership×s2: 42072 (eval 42073)
- persona×s3:    42074 (eval 42075)
- membership×s3: 42076 (eval 42077)
- persona×s1:    42078 (eval 42079)
- membership×s1: 42080 (eval 42081)
- persona×s5:    42082 (eval 42083)
- membership×s5: 42084 (eval 42085)
- persona×s6:    42086 (eval 42087)
- membership×s6: 42088 (eval 42089)
- persona×s11:   42090 (eval 42091)
- membership×s11: 42092 (eval 42093)
- persona×s12:   42094 (eval 42095)
- membership×s12: 42096 (eval 42097)

Queue after: 1 running + 71 pending = 72 total.
- 1 R + 1 PD eval = persona × 2b-it × s4 (in flight, walltime=9h)
- 34 PD = the new 2b-it cells at walltime=5h
- 36 PD = 9b-it cells at walltime=9h (untouched)

Note: only walltime was reduced; CPU=4 and MEM=64G are unchanged. The
2b-it × ifeval-concat MEM=96G case is no longer in scope (cancelled).
