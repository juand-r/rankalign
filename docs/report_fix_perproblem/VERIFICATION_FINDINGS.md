# report_fix.tex — verification findings (2026-06-24)

Thorough re-check after repeated mistakes. Every number independently recomputed; ground truth
checked instead of assumed.

## PASS — independently verified
1. **All 4 headline tables** (Q1 gen ROC, concordance, Spearman, Pearson) — `verify_report_fix.py`
   recomputes each per-problem metric from raw with a FRESH implementation and diffs vs the
   committed `fix_*.tex`. Result: **ALL CHECKS PASS** (4 tables × 4 cells × 5 settings within 1.5e-3).
2. **File selection** — eyeballed the matched score file for all 16 (combo×setting) cells: correct
   model / task / setting-token / epoch in every case (e.g. gemma membership s4 =
   `d2.69-e2-membership...tcs`; s3 = same delta, no tc; ifeval s2 = `delta1.38-epoch2-ifeval`).
3. **Grouping** — every file groups into the expected unit (membership 68 categories / ifeval 79
   prompts) and every group has both classes.
4. **gemma_membership s1** absence is CORRECT, not a bug: there is genuinely no s1 membership
   train file (the original report also omitted it).
5. **train-vs-test** — both panels now present and recomputed:
   - membership TRAIN (68 cat) vs rosch TEST (10 cat): s2 .945/.895, s3 .947/.886, s4 .974/.926.
   - ifeval TRAIN (79 prompts) vs ifeval-OOD TEST (20 prompts): base .792/.783, s1 .646/.582,
     s2 .491/.434, s3 .841/.785, s4 .837/.806.
   Test means independently recomputed from the per-problem long tables — match exactly.
6. **Figures** — all per-problem (unified ROC + concordance, TC-dynamics, TC-comparison,
   score-delta, train-vs-test); regenerated from the same verified pipeline and visually checked
   (dynamics, concordance, train-vs-test). Scatter + per-category were already per-problem/raw.
7. **report_fix.tex** compiles clean: rc=0, 0 undefined refs, 0 missing figures, 19 pp, no stale
   pooled `plots/` includes; superseded single-panel table removed.
8. **report_humaneval.tex** compiles clean: 15 pp, 0 undefined/missing/TODO.
9. **Environment**: 0 mll jobs queued; 0 RunPod pods (personal AND TAUR). git in sync with origin.

## Mistakes found and FIXED during this audit
- **False "zero ifeval-OOD files" claim.** The OOD panel data is in
  `metrics-from-scores/ifeval_v7_ood_9b-it_gen_roc_table_long.csv` (per-prompt), built from the
  10,953 raw `ifeval-prompt_*_test` score files. Earlier glob was wrong. → ifeval panel now built.
- **mll silently 1 commit behind.** A `git pull` reported "Updating de61fddb..77bc8920" but
  ABORTED on an untracked file; I'd grepped the output and never checked HEAD, so the new figure
  code never reached mll and `fix_train_vs_test.png` was never generated. The rsync "broken pipe"
  was actually **source-file-not-found**, not a network problem. → verified HEAD == target,
  regenerated, downloaded, recompiled.
- **s1 missing from rosch panel** — not a bug (no s1 membership-train file); builder now general.

## KNOWN, DOCUMENTED LIMITATION (not recoverable without re-eval)
- **rosch s5/s11**: the original report's s5/s11 rosch-test came from
  `metrics-from-scores/rosch_v7_...` which no longer exists; only `rerun-only` (s1–s4, s7) survives,
  and raw s5/s11 rosch-test scores were not retained. So s5/s11 are omitted from the membership
  panel and noted in the report caption. Recovering them needs a re-eval of s5/s11 on rosch.

## Process lessons (to avoid repeats)
- Verify HEAD after every pull; never trust grep'd pull output.
- "Broken pipe" on rsync → check the source file EXISTS before blaming the network.
- Check ground truth; don't conclude ("network", "zero files") from an unverified inference.
