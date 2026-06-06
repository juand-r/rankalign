# TASK 3 — Paper results that are NOT epoch2 (should be)

**Compiled 2026-06-02.** Which models/evals reported (or staged) for the paper were evaluated
at **epoch0 or epoch1** instead of the intended **epoch2 (final)**, and what it takes to fix each.

**Method:** evaluated epoch read authoritatively from the `file` column of
`metrics-from-scores/*_table_long.csv` (the exact `scores_` file that fed each table cell),
cross-checked against `v7/eval_inventory_v7.md`, `v7/training_inventory_v7.md`, and HF repos.
Confidence is marked per row.

---

## TL;DR — the non-epoch2 cells

| # | Model | Task | Setting | Evaluated at | e2 model exists? | Fix |
|---|-------|------|---------|--------------|------------------|-----|
| 1 | gemma-4-31B-it | humaneval correct-upper | **s2 RankAlign** | **epoch1** | ✅ yes (latkes) | **eval-only** — run e2 eval |
| 2 | gemma-4-31B-it | humaneval correct-upper | **s1 SFT-lo** | **v6, epoch1** | ❌ no v7 model | train v7 → eval, or accept v6 |
| 3 | gemma-4-31B-it | humaneval correct-upper | **s13 SFT+cft** | **epoch0** | ❌ e1/e2 never trained | resume/retrain → eval |
| 4 | qwen3.5-9b | ifeval | **s13 SFT+cft** | **epoch0** | ❌ only e0 | retrain to e2 → eval |
| 5 | qwen3.5-9b | membership | **s13 SFT+cft** | **epoch0** | ❌ only e0 | retrain to e2 → eval |

**The pattern:** **s13 (SFT + consistency-ft) is epoch-incomplete everywhere** — it was added
late (2026-05-24) and ran out of training time, so it only ever reached epoch0. And in the
headline **gemma-4 correct-upper** table, **RankAlign (s2)** and **SFT (s1)** are the two
non-epoch2 cells.

---

## What IS epoch2 (verified clean — no action needed)

From the final `*_table_long.csv` files (`gen_roc`), evaluated epoch per method:

- **gemma-4-31B-it correct-multi (cm):** RankAlign, New+PMI+fsx, New+NegTC+fsx, SFT — **all epoch2** ✅
- **gemma-4-31B-it correct-upper (cu):** New+fsx (s3), New+PMI+fsx (s4), New+NegTC+fsx (s7) — **all epoch2** ✅ (only s2/s1 below are not)
- **persona-v1** — gemma-2-2b, gemma-2-2b-it, gemma-2-9b-it, all of {all, id, ood} splits, every method — **all epoch2** ✅
- **rosch / membership** — gemma-2-9b-it, every method — **all epoch2** ✅

> Note: the **stale** `eval_coverage_matrix.md` (May-24 snapshot) showed several persona/membership
> gemma-2-9b-it cells at `done(e0)`/`done(e1)`. Those were **re-evaluated at e2** before the final
> tables were built — the `table_long` file columns all say epoch2. So the coverage matrix is
> superseded; trust the metrics tables.

---

## Per-item detail + fix

### 1. gemma-4-31B-it · correct-upper · s2 RankAlign — evaluated at **epoch1**  [HIGH confidence]
- **Evidence:** every cell in `humaneval_v2.1correct-upper_g4-31B-it_gen_roc_table_long.csv` for
  method=RankAlign cites `…-delta2.14-epoch1--…` files. Headline table
  `docs/humaneval_cu_v7_tables_2026-05-25.md` row 2 = these epoch1 numbers
  (GenROC Raw 90.65, PMI base 87.58, Neg base 89.15 n=79).
- **Why epoch1:** epoch1 was treated as the intended RankAlign checkpoint; no time to eval e2.
- **e2 model:** EXISTS — `latkes/rankalign-v7-g4-31b-d214-e2-cu-all-sm0.1-fix1` (private; e0/e1/e2 all
  byte-verified, see `../provenance-cu-s2-rankalign/`).
- **Fix (eval-only, no training):** run the 82-task eval on the **e2** adapter in 4 modes
  (self/neg × bt0/bt1) via `scripts/setup_and_eval_s2ep1.sh` logic with `EP=2`. Also re-do the
  3 failing `basetypneg` tasks (humaneval_1/10/100) that made Neg base n=79.

### 2. gemma-4-31B-it · correct-upper · s1 SFT-lo — **v6 model, epoch1**  [HIGH confidence]
- **Evidence:** `docs/humaneval_cu_s1_sft_v6_tables_2026-05-25.md` (explicit "v6 delta0.15-epoch1"
  caveat); scores in `outputs_gemma4_from_pod-v7/s1_sft_v6/` are `v6-…-epoch1` (PMI base + Neg base
  only; no self/neg variants).
- **No v7 s1 model exists** for cu (confirmed: absent from `v7/training_inventory_v7.md`).
- **Fix:** train v7 s1 (SFT-lo, 3 epochs) on cu, then eval e2 (self+neg). OR accept the v6 epoch1
  number with the caveat (delta doesn't affect a pure-SFT run, but it's a different code generation).

### 3. gemma-4-31B-it · correct-upper · s13 SFT+cft — **epoch0**  [HIGH confidence]
- **Evidence:** `docs/humaneval_cu_s13_ep0_tables_2026-05-25.md`; scores in
  `outputs_gemma4_from_pod-v7/s13_ep0/` are all `epoch0`; only e0 adapter exists
  (`latkes/rankalign-v7-gemma-4-31B-it-d2.14-e0-…-cft-lo0.1-fix1`, also TAUR-dev, **public**).
- **Why:** training crashed on mll after epoch0 (1 checkpoint only).
- **Fix:** resume/retrain s13 on cu to epoch2 (the run is reproducible via `_overnight_launch.sh
  humaneval gemma-4-31B-it s13`), then eval. Only self+neg eval modes were run even at e0 (PMI
  base/Neg base dropped) — redo full eval at e2.

### 4–5. qwen3.5-9b · ifeval & membership · s13 SFT+cft — **epoch0**  [MED-HIGH confidence]
- **Evidence:** only e0 models on HF (`latkes/rankalign-v7-qwen3.5-9b-{ifeval,membership}-s13-ep0`);
  launched as 1-epoch jobs (`docs/qwen_s13_provenance_2026-05-25.md`). In `pod-results-*.md` the
  s13 row is mostly `–`/`--` (not yet in the main tables).
- **Fix:** retrain qwen3.5-9b s13 (ifeval-concat, membership-sans-rosch-v0) to epoch2, then eval.

---

## Adjacent issues (not epoch, but "should fix" for the same cells)

- **gemma-4 cu partial columns:** s2 Neg base **n=79/82** (humaneval_1/10/100 fail in `basetypneg`
  mode); s4/s7/s11/s12 show **n=81/82**; cm s12 Neg base only **46/82**. These are missing-task
  gaps, fixable by re-running the few failed tasks (see `TASK2_eval_inventory.md §3`).
- **qwen3.5-9b s2 in-domain eval missing** — OOD done, ID never run; the trained s2 model was lost
  in pod migration (memory `qw35_s2_eval_status`). Separate from the epoch issue.
- **gemma-2-9b-it ifeval s13** — not run at all (the `s13` row is `–` in pod-results ifeval).

---

## Remaining confirmation items (datastor1 second pass done 2026-06-02)
- datastor1 is back; it did **not** change the epoch findings (it holds only the v6 archive — no
  v7/gemma-4/qwen). The core TASK-3 answers (RankAlign cu = epoch1; s13 = epoch0) stand.
- ifeval and qwen cells use the `eval_model_sN` eval scheme (epoch not in the filename). The HF
  source repos for **gemma-2-9b-it (ra9b) ifeval are all `-ep2`**, so those evals are epoch2; the
  exact downloaded epoch per **qwen** cell is the one piece still on the (stopped) RunPod volumes
  — `[PENDING — pod volume]`, not datastor1.
- Whether s1/s13 (cu) supplementary tables actually appear in the submitted PDF vs the main
  4-method table — confirm against the paper source. The main `metrics-from-scores` cu table has
  only Base + RankAlign + New{,+PMI,+NegTC}+fsx.
