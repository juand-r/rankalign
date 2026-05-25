# Pod Tracking — rankalign v7 training run

Last updated: 2026-05-25 ~02:40 UTC

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
| ra-9b-ifeval-s1 | ifeval | s1 | 103.207.149.80:14004 | 🔄 TRAINING | 0/40 | Step 3397/5110 (66%) |
| ra-9b-ifeval-s2 | ifeval | s2 | 64.247.201.40:11878 | 🔄 EVAL | 16/40 | At prompt_12; re-running all 20 tasks self+neg TC |
| ra-9b-ifeval-s3 | ifeval | s3 | 64.247.201.40:19766 | 🔄 EVAL | 15/40 | At prompt_13; re-running all 20 tasks self+neg TC |
| ra-9b-ifeval-s4 | ifeval | s4 | 216.243.220.227:16449 | 🔄 EVAL | 16/20 | At prompt_12; self-TC only |
| ra-9b-ifeval-s7 | ifeval | s7 | 216.243.220.227:14628 | 🔄 EVAL | 15/20 | At prompt_11; neg-TC only |

### Gemma-2 persona+member — tables computed ✅
Stored at: `mll:/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/ra9b_persona_member/`
Table builder: `scripts/_build_ra9b_persona_member_table.py`

### Gemma-2 ifeval — pending
Tables not yet built (eval still running). Will need a new builder `_build_ra9b_ifeval_table.py`.

**Bug fixed 2026-05-25:** `run_gemma2_cell.sh` hardcoded `seq 1 21` for ifeval tasks;
prompt_14 has no data file. Fixed to read actual data files from `fixed-prompts-ifeval/`,
filter N≤21, sort numerically → 20 real test tasks. All 4 running eval pods were
restarted with the fix; s1 will use it automatically when training completes.

---

## Qwen3.5-9B pods (TAUR account)

| Pod | Dataset | Setting | IP:Port | Status | CSVs | Notes |
|-----|---------|---------|---------|--------|------|-------|
| qw35-person-s1 | persona | s1 | 64.247.201.52:11729 | 🔄 TRAINING | 0/12 | Step 4193/5110 (82%) |
| qw35-person-s2 | persona | s2 | 87.120.211.210:19946 | ✅ DONE | 12/12 | Downloaded to mll |
| qw35-person-s4 | persona | s4 | 103.207.149.99:18517 | 🔄 TRAINING | 0/6 | Step 4734/5110 (93%) |
| qw35-person-s7 | persona | s7 | 103.207.149.99:12307 | 🔄 TRAINING | 0/6 | Step 4542/5110 (89%) |
| qw35-member-s1 | membership | s1 | 103.207.149.99:12306 | 🔄 TRAINING | 0/20 | Step 4673/5110 (91%) |
| qw35-member-s2 | membership | s2 | 103.207.149.87:18679 | ✅ DONE | 20/20 | Downloaded to mll |
| qw35-member-s4 | membership | s4 | 103.207.149.87:15807 | 🔄 TRAINING | 0/10 | Step 4696/5111 (92%) |
| qw35-member-s7 | membership | s7 | 64.247.201.47:18203 | ✅ DONE | 10/10 | Downloaded to mll |
| qw35-ifeval-s1 | ifeval | s1 | 64.247.201.47:11402 | 🔄 TRAINING | 0/40 | Step 3703/5110 (72%) |
| qw35-ifeval-s2 | ifeval | s2 | 64.247.201.40:18256 | 🔄 TRAINING | 0/40 | Step 2734/5110 (53%) |
| qw35-ifeval-s4 | ifeval | s4 | 103.207.149.154:12996 | 🔄 TRAINING | 0/20 | Step 5023/5111 (98%) |
| qw35-ifeval-s7 | ifeval | s7 | 103.207.149.154:13621 | 🔄 TRAINING | 0/20 | Step 4875/5111 (95%) |

### QW35 downloads pending
3 pods done but CSVs not yet downloaded to mll:
- qw35-person-s2 (12 CSVs)
- qw35-member-s2 (20 CSVs)
- qw35-member-s7 (10 CSVs)

Download target on mll: `/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/qw35_persona_member/`

---

## TODO queue

- [x] Download qw35-person-s2, member-s2, member-s7 CSVs to mll (42 CSVs, done 2026-05-25)
- [ ] When qw35-person-s1/s4/s7 finish eval → download
- [ ] When qw35-member-s1/s4 finish eval → download
- [ ] When gemma-2-ifeval-s1/s2/s3/s4/s7 finish eval → download
- [ ] When qw35-ifeval-s4/s7 finish training → they'll auto-start eval
- [ ] When qw35-ifeval-s1/s2 finish training → they'll auto-start eval
- [ ] Build ifeval table builder (new script) for gemma-2 and qw35
- [ ] Build qw35 persona+member table builder (or reuse/extend existing one)
- [ ] Download script for qw35 pods (analogous to download_scores_ra9b_persona_member.sh)
