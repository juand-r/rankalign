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
