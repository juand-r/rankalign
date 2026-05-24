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
