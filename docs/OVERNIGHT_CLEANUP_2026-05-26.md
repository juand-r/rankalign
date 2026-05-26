# Overnight pod backup + cleanup — 2026-05-26 (autonomous)

Task: back up anything unsaved on every pod (models → HF, results/logs → repo), then stop pods that
are 100% backed up. **Read the "ACTION NEEDED FROM YOU" section first.**

## ⚠️ BLOCKER: latkes private HF storage is FULL
Uploads started failing with `403 Forbidden: Private repository storage limit reached`. Five model
uploads succeeded before the limit; the rest (intermediate epochs + the membership-s1 deliverable)
**could not be uploaded.** You'll need to free latkes private storage (delete/make-public some repos,
or upgrade the plan) to finish backing up the remaining items.

## ACTION NEEDED FROM YOU
1. **Free latkes private storage**, then upload the items below that aren't on HF yet.
2. **`membership-s1-ep2` (the rosch-s1 model) is NOT on HF** — its latkes repo is empty (upload
   failed). It IS safe on the **laptop**: `/home/jdr/hf_upload_tmp/membership-s1/` (10 shards, 17 GB,
   complete). Please upload it once storage is freed. It's also still on pods ytdj36 + 91ja5g3.
3. **Decide on intermediate epochs** (ep0/ep1). Your convention is ep2-only (every run kept only the
   final), so they're probably disposable — but I left their pods running rather than lose them.
   Tell me to stop those pods, or free storage so I can upload the intermediates first.
4. **Check pod `440v4r9mca9wfy` (v7b-ifeval-s1)** — it is actively TRAINING but looks like it may be
   a restart-loop (see below). Confirm it's intended or kill it.

## What IS backed up
**On latkes HF (verified, 18 GB / 10 shards each):**
- `rankalign-v7-qwen3.5-9b-ifeval-s1-ep2`
- `rankalign-v7-qwen3.5-9b-ifeval-s2-ep2`, `-ifeval-s2-ep1`  (fills the previously-missing qwen ifeval-s2 gap)
- `rankalign-v7-qwen3.5-9b-ifeval-s13-ep0`
- `rankalign-v7-qwen3.5-9b-membership-s13-ep0`

**On TAUR-dev HF (pre-existing):** `rankalign-v7-qwen3.5-9b-membership-s7-ep2` (+ membership-s2/s4, persona-s2).
gemma-2-9b-it ifeval eval targets `rankalign-v7-gemma2-9b-it-ifeval-s{1..7}-ep2`.

**On laptop only (NOT yet on HF):** `membership-s1-ep2` → `/home/jdr/hf_upload_tmp/membership-s1/`.

**Results:** all qwen (ifeval s1/s13, rosch s1/s13) + gemma (s1–s7 ID) scored CSVs in the repo + pushed.
**Training logs:** qwen s1/s13 in `docs/qwen_model_uploads_2026-05-26/`; qwen s2 log inside the HF s2
repos; gemma s4 eval log in `docs/ra9b_id_eval_2026-05-25/`.

## What is NOT backed up (intermediate ep0/ep1 — on pods only, latkes full)
- `ifeval-s1-ep1`            → pod 6kjhz0xt5br82m (qw35-ifeval-s1)
- `membership-s7-ep0`, `-ep1` → pod ytdj36fjk5nouk (qw35-member-s7)
- `membership-s1-ep0`, `-ep1` → pod 91ja5g3th134sb (qw35-member-s1, 0-GPU CPU)
(These are training byproducts; your HF convention keeps only ep2. Left on pods pending your call.)

## Pod dispositions
**STOPPED (100% backed up):**
- `qrz3m6s1hm34ob` (gemma s4-migration) — s4 model is HF-cache symlink, results+v6 in repo, eval log saved.
- `xo5ntpx2v4qzce` (qw35-ifeval-s2-migration = the s13 pod) — all 4 of its models on latkes, results+logs in repo.

**LEFT RUNNING (your decision needed):**
- `6kjhz0xt5br82m` (qw35-ifeval-s1, H100, 64.247.201.47:11402) — deliverable ifeval-s1-ep2 on HF; holds intermediate ifeval-s1-ep1 (unbacked).
- `ytdj36fjk5nouk` (qw35-member-s7, H100, 64.247.201.47:12801) — deliverables backed (s7-ep2 HF, s1-ep2 laptop); holds ms7 ep0/ep1 intermediates + a copy of membership-s1-ep2 (not on HF).
- `91ja5g3th134sb` (qw35-member-s1, **0-GPU CPU, cheap**, 103.207.149.99:19550) — holds membership-s1 ep0/ep1/ep2 (ep2 = deliverable, on laptop but not HF).
- `440v4r9mca9wfy` (v7b-ifeval-s1, A40, 194.68.245.111:22064) — **UPDATE ~08:23 CT: now EXITED on its own.** It was training (step ~13/5110, no merged model) and stopped itself before completing (crashed, or RunPod reclaimed it). Could not have finished training in the interim, so **no merged model was produced — nothing to back up.** Pod is off so I can't inspect it; its volume persists if restarted. The v7b-ifeval-s1 training did NOT complete — **check whether you wanted this run** (it looked like a possible cell restart-loop that never produced a model). I did not restart it.

All gemma ID-eval pods (s1/s2/s3/s7) were already EXITED.

## Monitors / crons
- ra9b monitor crontab + ra9b heartbeat: REMOVED (gemma work done).
- `monitor_v7b_pods.sh` (:7/:37) and `monitor_qwen35_taur.sh` (:17/:47): still in crontab but **BENIGN** — neither has restart logic (they only check status / stop completed pods). Most of their target pods are EXITED. Left in place; safe to remove anytime.
- Heartbeat `4f552cd7` (:18/:48) is supervising and will keep this file updated.

## Cleanup done
- Deleted 5 empty failed latkes repos (the intermediate uploads that 403'd): ifeval-s1-ep1, membership-s7-ep0/ep1, membership-s1-ep0/ep1.
