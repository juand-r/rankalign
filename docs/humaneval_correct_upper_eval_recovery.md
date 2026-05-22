# humaneval-v2.1correct-upper — Pod Recovery Status

**Run date:** 2026-05-22  
**Pods stopped:** 10:57 UTC (TAUR Lab budget enforcement — 14 min early)  
**CSVs retrieved:** 0/3 settings — volumes intact, recovery needed

---

## Pod → Task Mapping

| Pod ID | Pod Name | Setting | Setting Name | Training | Epoch-2 adapter | Eval status |
|---|---|---|---|---|---|---|
| `iqd0qkpsqc8p81` | cu-s1-sft | **s1** | SFT-lo | epoch0 ✓ epoch1 ✓ epoch2 **?** (was ~79% at stop) | Uncertain | Never started |
| `jwe6rd8mrzstyd` | cu-s4-new-fsx-tc | **s4** | New+fsx+tc | All 3 epochs ✓ | ✓ saved 10:03 UTC | Partial: ~18–22/82 tasks done |
| `a49bizsicavwr0` | cu-s7-new-fsx-negtc | **s7** | New+fsx+negtc | All 3 epochs ✓ | ✓ saved ~10:10 UTC | Partial: ~42–55/82 tasks done |

Previously done (no recovery needed): **s2, s6, s9** — CSVs already in `trained_eval_outputs_3epoch/`.

---

## What's on Each Pod Volume

Volumes persist after pod stop. Restart on any GPU to recover.

### s1 — `iqd0qkpsqc8p81`
- **Adapters:** `/workspace/models_g4it/v6-google--gemma-4-31B-it-delta0.15-epoch{0,1}--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1`
  - epoch0 ✓, epoch1 ✓, epoch2: check on restart
- **CSVs:** 0 (eval never started)
- **Eval modes:** `--self-typicality` and `--neg-typicality` (both, per s1 config)
- **Recovery:** need to finish epoch-2 training first, then full 82-task eval (2 modes × 82 = 164 tasks)

### s4 — `jwe6rd8mrzstyd`
- **Adapter:** `/workspace/models_g4it/v6-google--gemma-4-31B-it-delta0.15-epoch2--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1`
- **CSVs on volume:** `scores_basetyp-*20260522.csv` — estimated ~18–22 files
- **Done markers:** `/workspace/outputs/.done/s4_ep2_self-typicality_bt1_*.done` — guards against re-running finished tasks
- **Eval mode:** `--self-typicality --base-typicality` (1 mode × 82 tasks)
- **Remaining:** ~60–64 tasks at ~3 min/task on H100 SXM ≈ 3h

### s7 — `a49bizsicavwr0`
- **Adapter:** `/workspace/models_g4it/v6-google--gemma-4-31B-it-delta0.15-epoch2--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1`
- **CSVs on volume:** `scores_basetypneg-*20260522.csv` — estimated ~42–55 files
- **Done markers:** `/workspace/outputs/.done/s7_ep2_neg-typicality_bt1_*.done`
- **Eval mode:** `--neg-typicality --base-typicality` (1 mode × 82 tasks)
- **Remaining:** ~27–40 tasks at ~1 min/task on H100 SXM ≈ 40 min

---

## Recovery Steps (per pod)

```bash
source /home/jdr/racas-more/llm-consistency-raca/.tools-venv/bin/activate
HF_TOKEN=$(python3 -c "from key_handler import KeyHandler; KeyHandler.set_env_key(); import os; print(os.environ['HF_TOKEN'])")
TAUR_KEY=$(python3 -c "from key_handler import KeyHandler; KeyHandler.set_env_key(); import os; print(os.environ['RUNPOD_API_KEY_TAUR'])")
SSH="ssh -i /home/jdr/.runpod/ssh/RunPod-Key-Go -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

# 1. Restart pod (UI or API) — get new SSH info:
read ip port <<< $(RUNPOD_API_KEY="$TAUR_KEY" runpodctl pod get <POD_ID> | python3 -c \
  "import sys,json; d=json.load(sys.stdin); s=d.get('ssh',{}); print(s.get('ip',''),s.get('port',''))")

# 2. Resume eval (idempotent — done markers skip already-finished tasks):
$SSH -p $port root@$ip "
  export HF_TOKEN='$HF_TOKEN'
  nohup bash /workspace/rankalign/pod-setup-train-scripts-gemma-4/run_settings_v21correct_upper.sh <SETTING> \
    >> /workspace/logs/run_s<SETTING>_restart.log 2>&1 < /dev/null &
  disown; echo PID=\$!
"

# 3. Wait for SETTING<N>_COMPLETE in log, then rsync:
LOCAL_DIR="/home/jdr/racas-more/llm-consistency-raca/notes/log_P_diff_plots/humaneval-v2.1correct-upper/trained_eval_outputs_3epoch"
$SSH -p $port root@$ip "cd /workspace/outputs && ls scores_*.csv | tar czf - -T -" \
  | tar xzf - -C "$LOCAL_DIR/"

# 4. Rsync to mll:
rsync -az "$LOCAL_DIR"/scores_*s<SETTING>*.csv \
  jdr@slurm-submit.cs.utexas.edu:/datastor2/jdr/rankalign/outputs_gemma4_from_pod/

# 5. Git commit + push:
cd /home/jdr/racas-more/llm-consistency-raca
git add notes/log_P_diff_plots/humaneval-v2.1correct-upper/trained_eval_outputs_3epoch/scores_*s<SETTING>*.csv
git commit -m "correct-upper s<SETTING>: add eval score CSVs (recovery run)"
git push

# 6. Stop pod (confirm with user first).
```

---

## Budget Estimate for Recovery (TAUR Lab)

H100 SXM rate on TAUR: ~$13/hr/pod (based on $39.50/hr for 3 pods during training run).

| Setting | Tasks remaining | Est. time | Est. cost (H100) | Cost (A100 80GB ~$2.50/hr) |
|---|---|---|---|---|
| s7 | ~27–40 | ~40 min | **~$9** | ~$2 |
| s4 | ~60–64 | ~3 h | **~$39** | ~$8–10 |
| s1 | 82 train + 164 eval | ~8–10 h | **~$110+** | ~$25 |

With $35 TAUR balance: **s7 fits; s4 does not on H100** (over by ~$4). Both fit on A100 80GB if available on TAUR.

---

## Known Monitor Script Bug

`monitor_v21correct_upper.sh` checks `ls scores_*s${setting}*.csv` but actual filenames are:
- s4: `scores_basetyp-*` (no literal `s4` in name)
- s7: `scores_basetypneg-*` (no literal `s7` in name)

This caused the monitor to never detect eval completion and never trigger download+stop.
Fix before any future run using this script.
