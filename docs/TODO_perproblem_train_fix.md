# TODO — per-problem train-set scoring (2026-06-19)

## DONE / in progress
HumanEval train-set eval is being redone PER-PROBLEM (score each problem's own ~28 train
candidates, then average ±SE), replacing the earlier GLOBAL-pool train eval (GROUP B + the
cancelled trs2), which scored the whole train pool once and is NOT comparable to the
per-problem test metric. All HumanEval train-side report values (train grid, per-epoch
dynamics, train-vs-test, concordance/scatter/delta) are being recomputed.

## REMINDER for the user (raised 2026-06-19, user leaving):
The ORIGINAL report (analysis/report.tex, ifeval + hyponym/rosch, gemma-2-9b-it + qwen-3.5)
used GLOBAL train scoring too (run_trainset_dynamics.sh evaluated the whole train task in one
eval — "the whole 2000-item training set sits in L_train"). So it has the SAME flaw. We need a
NEW "fix" version of that original report that scores the train set PER-PROBLEM (per rosch
category / per ifeval prompt) and re-derives the train-side numbers. Surface this when the user
is back.
