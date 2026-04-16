# Eval Status -- Updated 2026-04-14 (verified against inventory_scores.py)

## INVENTORY SUMMARY: 79/89 complete, 10 incomplete, 641 missing task files

## CURRENTLY RUNNING

### IFEval tc-neg TRAINING (9b-it) -- 5 jobs, ~19.5h in
| Job   | Variant                              | Epoch |
|-------|--------------------------------------|-------|
| 27884 | comb semi 0.1, neg-typcorr, log-odds | 3/3   |
| 27885 | sft semi 0.1, neg-typcorr            | 2/3   |
| 27886 | comb labelonly 0.1, neg-typcorr, log-odds | 3/3 |
| 27887 | pref-only labelonly 0.1, neg-typcorr, log-odds | 3/3 |
| 27888 | pref-only labelonly 0.1, neg-typcorr  | 3/3   |

### V2G Baseline EVALS (re-running, models now in correct location)
| Job   | Model    | Domain      | Eval type |
|-------|----------|-------------|-----------|
| 28039 | 9b-it    | plausibleqa | neg-tc    |
| 28040 | 9b-it    | ambigqa     | neg-tc    |
| 28041 | 9b-it    | hypernym    | neg-tc    |
| 28043 | 9b-it    | plausibleqa | self-tc   |
| 28044 | 9b-it    | ambigqa     | self-tc   |
| 28045 | 9b-it    | hypernym    | self-tc   |
| 28047 | 2b       | plausibleqa | neg-tc    |
| 28048 | 2b       | ambigqa     | neg-tc    |
| 28049 | 2b       | hypernym    | neg-tc    |
| 28050 | 2b       | ifeval      | neg-tc    |
| 28051 | 2b       | plausibleqa | self-tc   |
| 28052 | 2b       | ambigqa     | self-tc   |
| 28053 | 2b       | hypernym    | self-tc   |
| 28054 | 2b       | ifeval      | self-tc   |

NOTE: 9b-it ifeval V2G evals (28042, 28046) were CANCELLED -- model was corrupt.

### Plain-trained gap-fill EVALS (running)
| Job   | Domain   | Variant                    |
|-------|----------|----------------------------|
| 28058 | hypernym | comb vallogodds labelonly   |
| 28059 | hypernym | comb vallogodds semi        |

### Retraining V2G ifeval-concat 9b-it
| Job   | What                                  |
|-------|---------------------------------------|
| 28064 | V2G ifeval-concat 9b-it retrain (pending, 2 GPU, 24h) |

## REMAINING GAPS (from inventory_scores.py)

### 9b-it Hypernym plain-trained: 2 variants partially missing
- **comb labelonly**: 12/18 done, missing: hammers, helmets, jackets, kayaks, kites, mirrors
- **comb semi**: 12/18 done, missing: hammers, helmets, jackets, kayaks, kites, mirrors

```bash
MSUF="_merged"
MH=../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion
TASKS_HYP="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MH}--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1${MSUF} --neg-typcorr --log-odds -- $TASKS_HYP
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MH}--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1${MSUF} --neg-typcorr --log-odds -- $TASKS_HYP
```
(skip logic will handle the 12 already-done tasks)

### Models that DO NOT EXIST (will stay at 0):
- 9b-it ambigqa plain `pref-only vallogodds labelonly`: 0/17
- 9b-it hypernym plain `pref-only vallogodds labelonly`: 0/18
- 9b-it ifeval plain `pref-only vallogodds labelonly`: 0/99

## TODO ONCE CURRENT JOBS FINISH

### 1. After training jobs 27884-27888 finish:
Run IFEval tc-neg EVALS for all 5 newly trained models (5 x 99 = 495 tasks).
Models will be in `rankalign/models/` (trained via `run_train_semi.sh` which cd's correctly).

```bash
MSUF="_merged"
TASKS_IFE="ifeval-prompt_1 ifeval-prompt_2 ifeval-prompt_3 ifeval-prompt_4 ifeval-prompt_5 ifeval-prompt_6 ifeval-prompt_7 ifeval-prompt_8 ifeval-prompt_9 ifeval-prompt_10 ifeval-prompt_11 ifeval-prompt_12 ifeval-prompt_13 ifeval-prompt_15 ifeval-prompt_16 ifeval-prompt_17 ifeval-prompt_18 ifeval-prompt_19 ifeval-prompt_20 ifeval-prompt_21 ifeval-prompt_22 ifeval-prompt_23 ifeval-prompt_24 ifeval-prompt_25 ifeval-prompt_26 ifeval-prompt_27 ifeval-prompt_28 ifeval-prompt_29 ifeval-prompt_30 ifeval-prompt_32 ifeval-prompt_33 ifeval-prompt_34 ifeval-prompt_35 ifeval-prompt_36 ifeval-prompt_37 ifeval-prompt_38 ifeval-prompt_39 ifeval-prompt_40 ifeval-prompt_41 ifeval-prompt_42 ifeval-prompt_43 ifeval-prompt_44 ifeval-prompt_45 ifeval-prompt_46 ifeval-prompt_47 ifeval-prompt_48 ifeval-prompt_49 ifeval-prompt_50 ifeval-prompt_51 ifeval-prompt_52 ifeval-prompt_53 ifeval-prompt_54 ifeval-prompt_56 ifeval-prompt_57 ifeval-prompt_58 ifeval-prompt_59 ifeval-prompt_61 ifeval-prompt_63 ifeval-prompt_64 ifeval-prompt_65 ifeval-prompt_66 ifeval-prompt_67 ifeval-prompt_68 ifeval-prompt_70 ifeval-prompt_72 ifeval-prompt_73 ifeval-prompt_74 ifeval-prompt_75 ifeval-prompt_76 ifeval-prompt_77 ifeval-prompt_78 ifeval-prompt_79 ifeval-prompt_80 ifeval-prompt_82 ifeval-prompt_83 ifeval-prompt_84 ifeval-prompt_85 ifeval-prompt_87 ifeval-prompt_88 ifeval-prompt_89 ifeval-prompt_90 ifeval-prompt_91 ifeval-prompt_92 ifeval-prompt_93 ifeval-prompt_94 ifeval-prompt_95 ifeval-prompt_96 ifeval-prompt_97 ifeval-prompt_98 ifeval-prompt_99 ifeval-prompt_100 ifeval-prompt_102 ifeval-prompt_103 ifeval-prompt_104 ifeval-prompt_105 ifeval-prompt_106 ifeval-prompt_107 ifeval-prompt_108 ifeval-prompt_109"

MI=../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--ifeval-concat-all--d2g--random--alpha1.0--tc-neg--full-completion

run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MI}--force-same-x--labelonly0.1${MSUF} --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MI}--force-same-x--vallogodds--labelonly0.1${MSUF} --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MI}--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1${MSUF} --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MI}--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1${MSUF} --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh ${MI}--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1${MSUF} --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
```

### 2. After retraining job 28064 finishes:
Verify model saved to `rankalign/models/` correctly, then run 2 evals:

```bash
MI9=../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--ifeval-concat-all--d2g--random--alpha1.0--full-completion_merged
run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$MI9" --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
run 1 8 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$MI9" --self-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE
```

### 3. Run the 2 hypernym gap-fill commands above (comb labelonly + comb semi, 6 missing each)

## KNOWN ISSUES

- `pref-only vallogodds labelonly` plain-trained model does NOT EXIST for ambigqa, hypernym, or ifeval (9b-it). These 3 configs (0/17, 0/18, 0/99) will stay at 0.
- `eval_by_claude.py` silently fails per-task when model can't load (exit 0, no scores). Always check stderr for OSError after eval jobs.
- V2G baseline scores show as "unclaimed" in inventory -- script doesn't track V2G baselines yet.
- The inventory only tracks `neg-` prefix evals. The `self-` prefix evals (8133 files) are all unclaimed.
