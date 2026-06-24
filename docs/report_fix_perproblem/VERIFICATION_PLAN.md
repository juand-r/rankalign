# report_fix.tex — thorough verification plan (2026-06-24)

Trigger: repeated mistakes (false "zero ifeval-OOD files" claim; s1 dropped from train-vs-test).
Goal: independently re-verify EVERY number, table, and figure in report_fix.tex — do not trust the
builder; recompute a second way and diff.

## Steps
1. **Finish train-vs-test**: embed both panels (membership↔rosch, ifeval↔ifeval-OOD) in the report;
   confirm s1 present; update prose + s5/s11 omission note; fix/replace the old pooled figure.
2. **Independent recompute** (verify_report_fix.py, run on mll, separate code path):
   - file selection: print the matched score file for every (combo, setting, ep2, self) — eyeball
     that model/task/setting/epoch tokens are correct (catch find_files_for_combo mis-selection).
   - grouping sanity: every file groups into the expected n (ifeval 79 prompts / membership 68
     categories) and every group has both classes for ROC.
   - recompute Q1 gen ROC, concordance, Spearman, Pearson per problem INDEPENDENTLY and diff
     against the committed fix_*.tex values (tolerance 1e-3).
   - train-vs-test: recompute test means from the long tables and diff vs the panel .tex.
3. **Cross-check shift direction** vs the original pooled numbers (membership ~unchanged; ifeval up)
   — sanity, not exact.
4. **Report integrity**: both reports compile, 0 undefined refs, no missing figures, no stale
   pooled `plots/` includes left in report_fix.tex; captions match the figures.
5. **HE report spot-check**: re-verify the §2.5 val-loss table + that its figures exist.
6. **Env**: git clean + in sync; no mll jobs / no pods running.
7. Write VERIFICATION_FINDINGS.md with every check + PASS/FAIL + evidence. Fix all FAILs.

## Status
- in progress 2026-06-24.
