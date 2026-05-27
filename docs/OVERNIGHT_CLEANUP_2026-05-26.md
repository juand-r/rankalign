# Pod backup + cleanup — 2026-05-26 (final state)

User approved making everything PUBLIC on latkes (public HF storage is free → bypasses the private
storage limit that had blocked uploads). All models from the pods + laptop uploaded as **public**.

## Models on latkes (ALL PUBLIC) — verified
| repo (latkes/rankalign-v7-qwen3.5-9b-…) | GB | status |
|---|---|---|
| ifeval-s1-ep1 | 17.9 | ✅ |
| ifeval-s1-ep2 | 18.0 | ✅ |
| ifeval-s2-ep1 | 18.0 | ✅ |
| ifeval-s2-ep2 | 18.0 | ✅ |
| ifeval-s13-ep0 | 18.0 | ✅ |
| membership-s1-ep2 | 17.9 | ✅ (uploaded from laptop) |
| membership-s13-ep0 | 18.0 | ✅ |
| membership-s7-ep0 | 18.0 | ✅ |
| membership-s7-ep1 | 18.0 | ✅ |
| membership-s7-ep2 | 18.0 | ✅ (also on TAUR-dev) |
| **membership-s1-ep0** | — | ⏳ uploading (CPU pod, slow) |
| **membership-s1-ep1** | — | ⏳ uploading (CPU pod, slow) |

Also on TAUR-dev: gemma2-9b ifeval eval targets `rankalign-v7-gemma2-9b-it-ifeval-s{1..7}-ep2`.
All results (qwen ifeval s1/s13/s2, rosch s1/s13) + training logs are in the repo (`longform`).

## Pods
**STOPPED (all backed up):**
- `qrz3m6s1hm34ob` (gemma s4-mig), `xo5ntpx2v4qzce` (qw35 s2/s13) — stopped earlier.
- `6kjhz0xt5br82m` (qw35-ifeval-s1) — STOPPED; ifeval-s1 ep1+ep2 on HF, results in repo.
- `ytdj36fjk5nouk` (qw35-member-s7) — STOPPED; membership-s7 ep0/1/2 + membership-s1-ep2 on HF, results in repo.
- `440v4r9mca9wfy` (v7b-ifeval-s1) — self-EXITED mid-training; never produced a model (nothing to back up). **Check whether that run was intended.**

**STILL RUNNING — `91ja5g3th134sb` (qw35-member-s1, 0-GPU CPU, cheap):**
Only remaining pod. It holds membership-s1 `ep0`/`ep1` (SFT-baseline intermediates) not yet on HF.
It has only **2 GB RAM** (so `upload_folder` OOM'd) and ~4 MB/s bandwidth, so a memory-frugal
**per-file** upload is grinding through ~34 GB (could take ~2 h). When it finishes I'll verify + stop
this pod. If it can't complete, these two SFT intermediates are the only thing that won't make it to
HF — the membership-s1 **deliverable (ep2) is already on HF**, so nothing important is at risk.

## Cleanup remaining (after the CPU-pod upload resolves)
- Stop `91ja5g3th134sb`.
- Tear down the benign monitor crontabs (`monitor_v7b_pods`, `monitor_qwen35_taur`) + the heartbeat cron.
