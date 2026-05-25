# Pod Tracking — rankalign v7 training run

Last updated: 2026-05-25 ~03:30 UTC

Two models × 3 datasets × 5 settings = 27 pods total.
- **Gemma-2-9b-it** (personal RunPod account, 15 pods)
- **Qwen3.5-9B** (TAUR Lab RunPod account, 12 pods — no s3)

Settings:
- **s1**: SFT label-only 10%
- **s2**: RankAlign (semi 10%, no fsx, no TC)
- **s3**: New + fsx, no TC (gemma-2 only)
- **s4**: New + fsx + self-TC
- **s7**: New + fsx + neg-TC

Expected CSV counts per cell:
- **persona** (6 tasks): s1/s2/s3 → 12 CSVs, s4 → 6, s7 → 6
- **membership** (10 tasks): s1/s2/s3 → 20 CSVs, s4 → 10, s7 → 10
- **ifeval** (20 tasks, prompt_14 missing): s1/s2/s3 → 40 CSVs, s4 → 20, s7 → 20

---

## Gemma-2-9b-it pods (personal account)

| Pod | Dataset | Setting | IP:Port | Status | CSVs | Notes |
|-----|---------|---------|---------|--------|------|-------|
| ra-9b-persona-s1 | persona | s1 | 103.207.149.109:17014 | ✅ DONE | 12/12 | Downloaded to mll |
| ra-9b-persona-s2 | persona | s2 | 64.247.201.48:16874 | ✅ DONE | 12/12 | Downloaded to mll |
| ra-9b-persona-s3 | persona | s3 | 103.207.149.153:17458 | ✅ DONE | 12/12 | Downloaded to mll |
| ra-9b-persona-s4 | persona | s4 | 87.120.211.205:17651 | ✅ DONE | 6/6 | Downloaded to mll |
| ra-9b-persona-s7 | persona | s7 | 87.120.211.205:17165 | ✅ DONE | 6/6 | Downloaded to mll |
| ra-9b-member-s1 | membership | s1 | 216.243.220.217:10986 | ✅ DONE | 20/20 | Downloaded to mll |
| ra-9b-member-s2 | membership | s2 | 216.243.220.217:18162 | ✅ DONE | 20/20 | Downloaded to mll |
| ra-9b-member-s3 | membership | s3 | 216.243.220.217:18163 | ✅ DONE | 20/20 | Downloaded to mll |
| ra-9b-member-s4 | membership | s4 | 103.207.149.80:13362 | ✅ DONE | 10/10 | Downloaded to mll |
| ra-9b-member-s7 | membership | s7 | 103.207.149.80:14005 | ✅ DONE | 10/10 | Downloaded to mll |
| ra-9b-ifeval-s1 | ifeval | s1 | 103.207.149.80:14004 | 🔄 EVAL | 13/40 | Training done; eval in progress (self+neg TC) |
| ra-9b-ifeval-s2 | ifeval | s2 | 64.247.201.40:11878 | 🔄 EVAL | ~49 | Running; self+neg TC |
| ra-9b-ifeval-s3 | ifeval | s3 | 64.247.201.40:19766 | 🔄 EVAL | ~47 | Running; self+neg TC |
| ra-9b-ifeval-s4 | ifeval | s4 | 216.243.220.227:16449 | ✅ DONE | 20/20 | Downloaded; HF upload in progress; model uploading |
| ra-9b-ifeval-s7 | ifeval | s7 | 216.243.220.227:14628 | 🔄 EVAL | ~33 | Running; neg-TC only |

### Gemma-2 persona+member — tables computed ✅
Stored at: `mll:/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/ra9b_persona_member/`
Table builder: `scripts/_build_ra9b_persona_member_table.py`

### Gemma-2 ifeval — s4 DONE, s1/s2/s3/s7 in eval
s4 CSVs: `outputs_gemma4_from_pod-v7/ra9b_ifeval/` (33 CSVs = 20 canonical + 13 old names)
s1/s2/s3/s7: eval still running.

### HF model uploads (as of 2026-05-25 03:30 UTC)
DONE: g2-persona s1/s2/s3/s4/s7, g2-membership s4, qw35-persona-s2, qw35-member-s2/s7
IN PROGRESS: g2-membership s1/s2/s3/s7, g2-ifeval-s4, qw35-member-s4

**Bug fixed 2026-05-25:** `run_gemma2_cell.sh` hardcoded `seq 1 21` for ifeval tasks;
prompt_14 has no data file. Fixed to read actual data files from `fixed-prompts-ifeval/`,
filter N≤21, sort numerically → 20 real test tasks. All 4 running eval pods were
restarted with the fix; s1 will use it automatically when training completes.

---

## Qwen3.5-9B pods (TAUR account)

| Pod | Dataset | Setting | IP:Port | Status | CSVs | Notes |
|-----|---------|---------|---------|--------|------|-------|
| qw35-person-s1 | persona | s1 | 64.247.201.52:11729 | 🔄 TRAINING | 0/12 | Epoch 2/3 |
| qw35-person-s2 | persona | s2 | 87.120.211.210:19946 | ✅ DONE | 12/12 | Downloaded; HF upload done |
| qw35-person-s4 | persona | s4 | 103.207.149.99:18517 | 🔄 TRAINING | 0/6 | Epoch 2/3 |
| qw35-person-s7 | persona | s7 | 103.207.149.99:12307 | 🔄 TRAINING | 0/6 | Epoch 2/3 |
| qw35-member-s1 | membership | s1 | 103.207.149.99:12306 | 🔄 TRAINING | 0/20 | Epoch 2/3 |
| qw35-member-s2 | membership | s2 | 103.207.149.87:18679 | ✅ DONE | 20/20 | Downloaded; HF upload done |
| qw35-member-s4 | membership | s4 | 103.207.149.87:15807 | ✅ DONE | 10/10 | Downloaded; HF upload in progress |
| qw35-member-s7 | membership | s7 | 64.247.201.47:18203 | ✅ DONE | 10/10 | Downloaded; HF upload done |
| qw35-ifeval-s1 | ifeval | s1 | 64.247.201.47:11402 | 🔄 TRAINING | 0/40 | Still training |
| qw35-ifeval-s2 | ifeval | s2 | 64.247.201.40:18256 | 🔄 TRAINING | 0/40 | Epoch 1/3 |
| qw35-ifeval-s4 | ifeval | s4 | 103.207.149.154:12996 | 🔄 MERGING | 0/20 | Training done; merging LoRA; eval starting soon |
| qw35-ifeval-s7 | ifeval | s7 | 103.207.149.154:13621 | 🔄 TRAINING | 0/20 | Epoch 1/3 |

### QW35 downloads pending
3 pods done but CSVs not yet downloaded to mll:
- qw35-person-s2 (12 CSVs)
- qw35-member-s2 (20 CSVs)
- qw35-member-s7 (10 CSVs)

Download target on mll: `/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/qw35_persona_member/`

---

## TODO queue

- [x] Download qw35-person-s2, member-s2, member-s7 CSVs to mll (42 CSVs, done 2026-05-25)
- [x] Download g2-ifeval-s4 CSVs (33 files, done 2026-05-25)
- [x] Download qw35-member-s4 CSVs (10 files, done 2026-05-25)
- [x] HF model uploads for g2-persona s1/s2/s3/s4/s7 — DONE
- [x] HF model uploads for g2-member s4 — DONE
- [x] HF model uploads for qw35-persona-s2, qw35-member-s2/s7 — DONE
- [ ] HF model uploads for g2-member s1/s2/s3/s7 — IN PROGRESS
- [ ] HF model uploads for g2-ifeval-s4 — IN PROGRESS
- [ ] HF model uploads for qw35-member-s4 — IN PROGRESS
- [ ] When g2-ifeval-s1/s2/s3/s7 finish eval → download + HF upload model + stop pod
- [ ] When qw35-ifeval-s4 finishes eval → download + HF upload model + stop pod
- [ ] When qw35-person-s1/s4/s7, qw35-member-s1 finish training → eval auto-starts → download + HF upload model + stop pod
- [ ] When qw35-ifeval-s1/s2/s7 finish training → eval auto-starts → same process
- [ ] Build ifeval table builder (new script) for gemma-2 and qw35
- [ ] Build qw35 persona+member table builder (or reuse/extend existing one)
- [ ] Download remaining CSVs to mll (once eval pods complete)
- [ ] Stop all done pods after model upload confirmed
