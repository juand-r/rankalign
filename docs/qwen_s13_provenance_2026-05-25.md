# Qwen3.5-9B s13 (SFT + consistency-ft) runs — provenance (2026-05-25)

Launched autonomously 2026-05-25 (~14:5x CT) to add the **s13** setting for Qwen3.5-9B,
which had no s13 results or models (verified: 0 s13 CSVs in pod dirs, 0 s13 repos in
TAUR-dev HF). gemma-2-9b-it ifeval s13 was NOT run here — a colleague already did it.

## What was launched
Two 1-epoch jobs, **fixed delta = 0.15**, run sequentially on one pod:
1. **ifeval s13** — train on `ifeval-concat`, OOD eval (prompts 1–21).
2. **membership s13** — train on `membership-sans-rosch-v0`, eval 10 rosch tasks.

s13 = SFT-lo + `--consistency-ft` (recipe from `_overnight_launch.sh`):
`--preference_loss_weight 0 --nll_validator_weight 1 --nll_generator_weight 1
--labeled-only 0.1`, **no** force-same-x, **no** train-time TC, **no** validator-log-odds;
eval modes self + neg + base-typicality. Model dir suffix:
`...--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1`.

## Reproduce
Committed scripts (branch `longform`):
- `scripts/run_qwen35_v7b_cell.sh DATASET s13` — qwen, fixed delta 0.15 (no --delta-bins), `EPOCHS` env override.
- `scripts/run_qwen35_v7b_s13_both.sh` — chains ifeval then membership at EPOCHS=1.

On the pod:
```bash
cd /workspace/rankalign && git fetch origin longform && git reset --hard origin/longform
cd scripts && HF_TOKEN=... nohup bash run_qwen35_v7b_s13_both.sh >> /workspace/logs/qwen_s13_both.log 2>&1 </dev/null &
```
Base model `Qwen/Qwen3.5-9B`; `ranking_loss_ref_fix.py`; `--lora`; transformers 5.8.1
with the moe.py torch-2.4.1 patch (applied by the cell script).

## Compute
- Pod **xo5ntpx2v4qzce** "qw35-ifeval-s2-migration" — **TAUR account**, 1× H100 SXM (80GB), $3.29/hr.
- Reused (was idle; its prior trained model had been lost, but the qwen env/deps/base-model cache remain). No EXITED pod restarted (avoids host-full risk).
- Pod has NO tmux (launched via nohup) and NO rsync (downloads use tar over ssh).

## Monitoring
- System cron `monitor_qwen_s13.sh` at `:23,:53` → on `V7B_IFEVAL_S13_DONE` / `V7B_MEMBERSHIP_S13_DONE`,
  tar-downloads `scores_*eval_model_s13_*` to `outputs_gemma4_from_pod-v7/qw35_ifeval` /
  `qw35_persona_member`, git commit + push. **Does NOT stop the pod.**
- In-session heartbeat CronCreate `8ed94a96` at `:08,:38` (restarts on crash; downloads if cron misses).

## Teardown (manual — pod NOT auto-stopped)
When both `qwen_s13_done/{ifeval_s13,membership_s13}` flags exist and CSVs are pushed:
stop pod `xo5ntpx2v4qzce` (TAUR: `RUNPOD_API_KEY="$RUNPOD_API_KEY_TAUR" runpodctl pod stop xo5ntpx2v4qzce`)
and remove the crontab: `crontab -l | grep -v monitor_qwen_s13 | crontab -`.

## Timing note
1-epoch train + eval. Will NOT finish by the midnight CT paper deadline — results expected
overnight / next morning. Tables rebuild via `scripts/_build_pod_morning_results.py` (the s13
rows will populate once CSVs land; setting detection keys on `--cft--`).
