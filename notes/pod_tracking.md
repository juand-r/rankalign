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
| ra-9b-persona-s1 | persona | s1 | 103.207.149.109:17014 | ✅ STOPPED | 12/12 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-persona-s1-ep2 |
| ra-9b-persona-s2 | persona | s2 | 64.247.201.48:16874 | ✅ STOPPED | 12/12 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-persona-s2-ep2 |
| ra-9b-persona-s3 | persona | s3 | 103.207.149.153:17458 | ✅ STOPPED | 12/12 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-persona-s3-ep2 |
| ra-9b-persona-s4 | persona | s4 | 87.120.211.205:17651 | ✅ STOPPED | 6/6 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-persona-s4-ep2 |
| ra-9b-persona-s7 | persona | s7 | 87.120.211.205:17165 | ✅ STOPPED | 6/6 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-persona-s7-ep2 |
| ra-9b-member-s1 | membership | s1 | 216.243.220.217:10986 | ✅ STOPPED | 20/20 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-membership-s1-ep2 |
| ra-9b-member-s2 | membership | s2 | 216.243.220.217:18162 | ✅ STOPPED | 20/20 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-membership-s2-ep2 |
| ra-9b-member-s3 | membership | s3 | 216.243.220.217:18163 | ✅ STOPPED | 20/20 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-membership-s3-ep2 |
| ra-9b-member-s4 | membership | s4 | 103.207.149.80:13362 | ✅ STOPPED | 10/10 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-membership-s4-ep2 |
| ra-9b-member-s7 | membership | s7 | 103.207.149.80:14005 | ✅ STOPPED | 10/10 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-membership-s7-ep2 |
| ra-9b-ifeval-s1 | ifeval | s1 | 103.207.149.80:14004 | 🔄 EVAL | 13/40 | Training done; crashed at prompt_14 (fixed); eval restarted 2026-05-25 ~03:52 UTC |
| ra-9b-ifeval-s2 | ifeval | s2 | 64.247.201.40:11878 | ✅ STOPPED | 40/40 | CSVs+model done; HF: rankalign-v7-gemma2-9b-it-ifeval-s2-ep2 |
| ra-9b-ifeval-s3 | ifeval | s3 | 64.247.201.40:19766 | 🔄 UPLOADING | 40/40 | CSVs done+committed; HF model upload in progress |
| ra-9b-ifeval-s4 | ifeval | s4 | 216.243.220.217:16449 | ✅ STOPPED | 20/20 | CSVs done; HF: rankalign-v7-gemma2-9b-it-ifeval-s4-ep2 |
| ra-9b-ifeval-s7 | ifeval | s7 | 216.243.220.227:14628 | 🔄 UPLOADING | 20/20 | CSVs done+committed; HF model upload in progress |

### Gemma-2 persona+member — tables computed ✅
Stored at: `mll:/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/ra9b_persona_member/`
Table builder: `scripts/_build_ra9b_persona_member_table.py`

### Gemma-2 ifeval — s2+s3+s4+s7 DONE, s1 restarted (~1.5h to go)
s4: `outputs_gemma4_from_pod-v7/ra9b_ifeval/` (33 CSVs = 20 canonical + 13 old); HF: rankalign-v7-gemma2-9b-it-ifeval-s4-ep2
s2: `outputs_gemma4_from_pod-v7/ra9b_ifeval/` (53 CSVs = 40 canonical + 13 old); HF: rankalign-v7-gemma2-9b-it-ifeval-s2-ep2
s3: `outputs_gemma4_from_pod-v7/ra9b_ifeval/` (51 CSVs = 40 canonical + 11 old); HF upload in progress
s7: `outputs_gemma4_from_pod-v7/ra9b_ifeval/` (33 CSVs = 20 canonical + 13 old); HF upload in progress
s1: eval restarted after prompt_14 crash; ~1.5h remaining; expected 40 CSVs when done

### HF model uploads (as of 2026-05-25 04:05 UTC)
DONE: g2-persona s1/s2/s3/s4/s7, g2-membership s4, g2-ifeval-s2, qw35-persona-s2, qw35-member-s2/s7
IN PROGRESS: g2-membership s1/s2/s3/s7, g2-ifeval-s3, g2-ifeval-s4, g2-ifeval-s7, qw35-member-s4

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
| qw35-ifeval-s4 | ifeval | s4 | 103.207.149.154:12996 | 🔄 TRAINING | 0/20 | Epoch 1/3 of training (~step 990/5111); ~13h remaining |
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
