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

## SPEC confirmed by user (2026-06-19)
- Per-problem train eval: use ALL candidates per train problem (NO subsample).
- That is ~15 positive + ~14 negative per problem (median; min 10 pos / 9 neg, max 15/15).
- 80 train problems (HumanEval/0,101,102,...); train/test are DISJOINT problem sets
  (82 test problems, 0 overlap) — a true held-out-problem generalization split.
- **MUST state these details in the report** (per-problem, all candidates, ~15 pos/14 neg,
  disjoint problem-level split, 80 train vs 82 test problems).
- Monitor (laptop crontab + CronCreate heartbeat fdbb9b38) turned OFF 2026-06-19; re-arm later.

## Duplicate rows in train.csv (process note — KEEP OUT OF THE REPORT, docs only)
- train.csv has pre-existing duplicate (task_id, answer, correct) rows: **165 upper / 37 multi**.
- Verified there is NO dedup anywhere in the path: load_csv_items, the humaneval loaders,
  eval_by_claude.py, and the trainer all keep them (the trainer's only `set()`/`unique` use is
  for unique *prompts* in the labeled/unlabeled semi-supervised split, not candidate rows).
- So they flow through training, eval, and the per-problem split unchanged. Using all
  candidates per problem (user decision) keeps them; within-problem duplicates score
  identically (redundant, harmless for ROC).
- **User directive (2026-06-19): do NOT mention this in report_humaneval.tex; this docs note
  is the record.**

## Validation (2026-06-19): per-problem train tasks WORK
- 160 tasks registered (humaneval-v2.1correct-{upper,multi}-train-<slug>, 80 each).
- humaneval-v2.1correct-multi-train-humaneval_101 -> L_test = 29 candidates, task_id HumanEval/101,
  14 pos / 15 neg. Problem-specific. (Run as NORMAL eval, no --train.)
