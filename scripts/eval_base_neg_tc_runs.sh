#!/bin/bash
# Base model (google/gemma-2-2b) evaluation with neg-typicality correction (LLR).
# Uses run_eval_semi.sh with --neg-typcorr flag.
# All tasks match the existing self-TC base eval set.
#
# 6 jobs total: hypernym, ambigqa, plausibleqa-nq, plausibleqa-trivia, plausibleqa-webq, ifeval

MODEL=google/gemma-2-2b

# ============================================================
# HYPERNYM (18 tasks, ~10 min)
# ============================================================
run 1 0 --min 55 scripts/run_eval_semi.sh $MODEL --neg-typcorr --log-odds -- \
    hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars \
    hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs \
    hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns \
    hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks \
    hypernym-kites hypernym-mirrors

# ============================================================
# AMBIGQA (17 tasks, ~15 min)
# ============================================================
run 1 0 --min 55 scripts/run_eval_semi.sh $MODEL --neg-typcorr --log-odds -- \
    ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives \
    ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays \
    ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used \
    ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year

# ============================================================
# PLAUSIBLEQA - NQ (33 tasks, ~30 min)
# ============================================================
run 1 2 scripts/run_eval_semi.sh $MODEL --neg-typcorr --log-odds -- \
    plausibleqa-nq_1114 plausibleqa-nq_1324 plausibleqa-nq_1328 plausibleqa-nq_1369 \
    plausibleqa-nq_1394 plausibleqa-nq_1438 plausibleqa-nq_1663 plausibleqa-nq_2031 \
    plausibleqa-nq_207 plausibleqa-nq_2174 plausibleqa-nq_2281 plausibleqa-nq_2421 \
    plausibleqa-nq_2436 plausibleqa-nq_2535 plausibleqa-nq_2622 plausibleqa-nq_2637 \
    plausibleqa-nq_2759 plausibleqa-nq_2824 plausibleqa-nq_2856 plausibleqa-nq_2867 \
    plausibleqa-nq_2876 plausibleqa-nq_3004 plausibleqa-nq_3015 plausibleqa-nq_3068 \
    plausibleqa-nq_3099 plausibleqa-nq_3127 plausibleqa-nq_3137 plausibleqa-nq_316 \
    plausibleqa-nq_3276 plausibleqa-nq_54 plausibleqa-nq_562 plausibleqa-nq_709 \
    plausibleqa-nq_958

# ============================================================
# PLAUSIBLEQA - TRIVIA (21 tasks, ~20 min)
# ============================================================
run 1 2 scripts/run_eval_semi.sh $MODEL --neg-typcorr --log-odds -- \
    plausibleqa-trivia_1655 plausibleqa-trivia_2984 plausibleqa-trivia_3035 \
    plausibleqa-trivia_3043 plausibleqa-trivia_3180 plausibleqa-trivia_3245 \
    plausibleqa-trivia_3433 plausibleqa-trivia_3492 plausibleqa-trivia_3599 \
    plausibleqa-trivia_4009 plausibleqa-trivia_4234 plausibleqa-trivia_4489 \
    plausibleqa-trivia_4697 plausibleqa-trivia_5003 plausibleqa-trivia_560 \
    plausibleqa-trivia_5675 plausibleqa-trivia_6317 plausibleqa-trivia_6777 \
    plausibleqa-trivia_7272 plausibleqa-trivia_7579 plausibleqa-trivia_9589

# ============================================================
# PLAUSIBLEQA - WEBQ (46 tasks, ~45 min)
# ============================================================
run 1 2 scripts/run_eval_semi.sh $MODEL --neg-typcorr --log-odds -- \
    plausibleqa-webq_1000 plausibleqa-webq_1046 plausibleqa-webq_1086 \
    plausibleqa-webq_1097 plausibleqa-webq_1163 plausibleqa-webq_1187 \
    plausibleqa-webq_1278 plausibleqa-webq_1307 plausibleqa-webq_1310 \
    plausibleqa-webq_1338 plausibleqa-webq_134 plausibleqa-webq_1383 \
    plausibleqa-webq_141 plausibleqa-webq_1421 plausibleqa-webq_1442 \
    plausibleqa-webq_1476 plausibleqa-webq_1498 plausibleqa-webq_15 \
    plausibleqa-webq_1584 plausibleqa-webq_1613 plausibleqa-webq_1668 \
    plausibleqa-webq_1714 plausibleqa-webq_1723 plausibleqa-webq_1836 \
    plausibleqa-webq_1972 plausibleqa-webq_212 plausibleqa-webq_299 \
    plausibleqa-webq_342 plausibleqa-webq_373 plausibleqa-webq_428 \
    plausibleqa-webq_435 plausibleqa-webq_520 plausibleqa-webq_611 \
    plausibleqa-webq_650 plausibleqa-webq_669 plausibleqa-webq_672 \
    plausibleqa-webq_713 plausibleqa-webq_744 plausibleqa-webq_749 \
    plausibleqa-webq_760 plausibleqa-webq_77 plausibleqa-webq_803 \
    plausibleqa-webq_84 plausibleqa-webq_88 plausibleqa-webq_882 \
    plausibleqa-webq_898

# ============================================================
# IFEVAL (13 tasks, ~20 min)
# ============================================================
run 1 2 scripts/run_eval_semi.sh $MODEL --neg-typcorr --log-odds --disc-shots-zero -- \
    ifeval-prompt_2 ifeval-prompt_3 ifeval-prompt_4 ifeval-prompt_6 \
    ifeval-prompt_7 ifeval-prompt_8 ifeval-prompt_10 ifeval-prompt_11 \
    ifeval-prompt_13 ifeval-prompt_15 ifeval-prompt_16 ifeval-prompt_20 \
    ifeval-prompt_21
