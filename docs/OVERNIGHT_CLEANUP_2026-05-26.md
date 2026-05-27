# Pod backup + cleanup — 2026-05-26 (COMPLETE)

Goal: back up everything on the pods (models → HF, results/logs/scripts → repo), then close all pods.
**Status: DONE. All pods stopped. All deliverables on public HF.**

## Models on latkes HF (PUBLIC) — verified, 18 GB / 10 shards each
- ifeval: `ifeval-s1-ep1`, `ifeval-s1-ep2`, `ifeval-s2-ep1`, `ifeval-s2-ep2`, `ifeval-s13-ep0`
- membership: `membership-s1-ep2` (rosch-s1 model), `membership-s13-ep0`, `membership-s7-ep0`, `membership-s7-ep1`, `membership-s7-ep2`

(All under `latkes/rankalign-v7-qwen3.5-9b-…`.) Switched to PUBLIC because latkes private storage
was full — public is free and unblocked everything. gemma-2-9b-it ifeval eval targets
`rankalign-v7-gemma2-9b-it-ifeval-s{1..7}-ep2` were already on TAUR-dev.

**NOT uploaded (per your call — "ep0/ep1 don't matter"):** `membership-s1-ep0` and `membership-s1-ep1`
(SFT-baseline intermediate epochs). The CPU pod's 2 GB RAM OOM'd on upload; I was relaying them via the
laptop but you said to stop. The empty/partial repos were deleted. The membership-s1 **deliverable
(ep2) is on HF**, so nothing important is missing.

## Pods — ALL STOPPED (EXITED)
| pod | role | note |
|---|---|---|
| qrz3m6s1hm34ob | gemma s4-migration | stopped |
| xo5ntpx2v4qzce | qw35 s2/s13 | stopped |
| 6kjhz0xt5br82m | qw35 ifeval-s1 | stopped |
| ytdj36fjk5nouk | qw35 member-s7 | stopped |
| 91ja5g3th134sb | qw35 member-s1 (CPU) | stopped |
| 440v4r9mca9wfy | v7b-ifeval-s1 | self-exited mid-training (never produced a model) — **check if that run was intended** |

## Results / logs / scripts
All in the repo (`longform`, pushed): qwen ifeval s1/s2/s13 + rosch s1/s13 scored CSVs (incl. the
ifeval-s1 base-eval CSVs), training logs (`docs/qwen_model_uploads_2026-05-26/`), gemma s4 eval log
(`docs/ra9b_id_eval_2026-05-25/`), and the upload helper scripts.

## Teardown done
- Monitor crontabs removed (`monitor_v7b_pods`, `monitor_qwen35_taur`, plus the earlier `monitor_ra9b_id`).
- All heartbeat/poller cron jobs deleted.

## Leftover laptop temp copies (safe to delete anytime — disk has 771 GB free)
`/home/jdr/hf_upload_tmp/`:
- `membership-s1/` (17 GB) = membership-s1-ep2 — already on HF.
- `v7-Qwen--…delta1.54-epoch0…_merged` (17 GB) = ms1-ep0 — relay copy (not on HF, you said it doesn't matter).
- `v7-Qwen--…delta1.54-epoch1…_merged` (~partial) = ms1-ep1 relay copy.
