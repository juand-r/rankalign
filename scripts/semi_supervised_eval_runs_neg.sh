#!/bin/bash
# Eval runs with --neg-typicality at eval time.
#
# Toggle MODEL to choose base model:
#   "gemma-2-2b"    = google/gemma-2-2b finetuned models
#   "gemma-2-9b-it" = google/gemma-2-9b-it finetuned models
#
# Toggle TC_SUFFIX to choose which finetuned models to evaluate:
#   "--tc-neg"  = models trained WITH neg-typicality
#   ""          = models trained WITHOUT any typicality correction (plain)
#
# Usage: source this file, or copy individual run commands.

# Toggle model
#MODEL="gemma-2-2b"
MODEL="gemma-2-9b-it"

# Toggle: "--tc-neg" for tc-neg trained models, "" for plain trained models
TC_SUFFIX="--tc-neg"
#TC_SUFFIX=""

# LoRA models need _merged suffix; 9b models need more eval time
if [[ "$MODEL" == *"9b"* ]]; then
    MSUF="_merged"
    H_SHORT=2   # ambigqa (17 tasks), hypernym (18 tasks)
    H_LONG=4    # plausibleqa (100 tasks)
    H_IFE=8     # ifeval (99 tasks, longer prompts)
else
    MSUF=""
    H_SHORT=2
    H_LONG=2
    H_IFE=4
fi

# ============================================================
# PLAUSIBLEQA (6 models × 100 tasks)
# ============================================================

TASKS_PQA="plausibleqa-nq_1114 plausibleqa-nq_1324 plausibleqa-nq_1328 plausibleqa-nq_1369 plausibleqa-nq_1394 plausibleqa-nq_1438 plausibleqa-nq_1663 plausibleqa-nq_2031 plausibleqa-nq_207 plausibleqa-nq_2174 plausibleqa-nq_2281 plausibleqa-nq_2421 plausibleqa-nq_2436 plausibleqa-nq_2535 plausibleqa-nq_2622 plausibleqa-nq_2637 plausibleqa-nq_2759 plausibleqa-nq_2824 plausibleqa-nq_2856 plausibleqa-nq_2867 plausibleqa-nq_2876 plausibleqa-nq_3004 plausibleqa-nq_3015 plausibleqa-nq_3068 plausibleqa-nq_3099 plausibleqa-nq_3127 plausibleqa-nq_3137 plausibleqa-nq_316 plausibleqa-nq_3276 plausibleqa-nq_54 plausibleqa-nq_562 plausibleqa-nq_709 plausibleqa-nq_958 plausibleqa-trivia_1655 plausibleqa-trivia_2984 plausibleqa-trivia_3035 plausibleqa-trivia_3043 plausibleqa-trivia_3180 plausibleqa-trivia_3245 plausibleqa-trivia_3433 plausibleqa-trivia_3492 plausibleqa-trivia_3599 plausibleqa-trivia_4009 plausibleqa-trivia_4234 plausibleqa-trivia_4489 plausibleqa-trivia_4697 plausibleqa-trivia_5003 plausibleqa-trivia_560 plausibleqa-trivia_5675 plausibleqa-trivia_6317 plausibleqa-trivia_6777 plausibleqa-trivia_7272 plausibleqa-trivia_7579 plausibleqa-trivia_9589 plausibleqa-webq_1000 plausibleqa-webq_1046 plausibleqa-webq_1086 plausibleqa-webq_1097 plausibleqa-webq_1163 plausibleqa-webq_1187 plausibleqa-webq_1278 plausibleqa-webq_1307 plausibleqa-webq_1310 plausibleqa-webq_1338 plausibleqa-webq_134 plausibleqa-webq_1383 plausibleqa-webq_141 plausibleqa-webq_1421 plausibleqa-webq_1442 plausibleqa-webq_1476 plausibleqa-webq_1498 plausibleqa-webq_15 plausibleqa-webq_1584 plausibleqa-webq_1613 plausibleqa-webq_1668 plausibleqa-webq_1714 plausibleqa-webq_1723 plausibleqa-webq_1836 plausibleqa-webq_1972 plausibleqa-webq_212 plausibleqa-webq_299 plausibleqa-webq_342 plausibleqa-webq_373 plausibleqa-webq_428 plausibleqa-webq_435 plausibleqa-webq_520 plausibleqa-webq_611 plausibleqa-webq_650 plausibleqa-webq_669 plausibleqa-webq_672 plausibleqa-webq_713 plausibleqa-webq_744 plausibleqa-webq_749 plausibleqa-webq_760 plausibleqa-webq_77 plausibleqa-webq_803 plausibleqa-webq_84 plausibleqa-webq_88 plausibleqa-webq_882 plausibleqa-webq_898"

MP=../models/v6-google--${MODEL}-delta0.15-epoch2--plausibleqa-all--d2g--random--alpha1.0${TC_SUFFIX}

# --- pref-only labelonly (no vallogodds) ---
run 1 $H_LONG --cpu 4 --mem 32G scripts/run_eval_semi.sh $MP--full-completion--force-same-x--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_PQA

# --- pref-only labelonly (vallogodds) ---
run 1 $H_LONG --cpu 4 --mem 32G scripts/run_eval_semi.sh $MP--full-completion--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_PQA

# --- pref-only semi ---
run 1 $H_LONG --cpu 4 --mem 32G scripts/run_eval_semi.sh $MP--full-completion--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_PQA

# --- comb labelonly ---
run 1 $H_LONG --cpu 4 --mem 32G scripts/run_eval_semi.sh $MP--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_PQA

# --- comb semi ---
run 1 $H_LONG --cpu 4 --mem 32G scripts/run_eval_semi.sh $MP--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_PQA

# --- sft semi ---
run 1 $H_LONG --cpu 4 --mem 32G scripts/run_eval_semi.sh $MP--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_PQA

# ============================================================
# AMBIGQA (5 epoch2 models × 17 tasks; pref-only semi at epoch1)
# ============================================================

TASKS_AQA="ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year"

MA=../models/v6-google--${MODEL}-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0${TC_SUFFIX}

# --- pref-only labelonly (no vallogodds) ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MA--full-completion--force-same-x--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_AQA

# --- pref-only labelonly (vallogodds) ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MA--full-completion--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_AQA

# --- pref-only semi ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MA--full-completion--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_AQA

# --- comb labelonly ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MA--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_AQA

# --- comb semi ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MA--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_AQA

# --- sft semi ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MA--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_AQA

# ============================================================
# HYPERNYM (6 models × 18 tasks)
# ============================================================

TASKS_HYP="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

MH=../models/v6-google--${MODEL}-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0${TC_SUFFIX}

# --- pref-only labelonly (no vallogodds) ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MH--full-completion--force-same-x--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_HYP

# --- pref-only labelonly (vallogodds) ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MH--full-completion--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_HYP

# --- pref-only semi ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MH--full-completion--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_HYP

# --- comb labelonly ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MH--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds -- $TASKS_HYP

# --- comb semi ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MH--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_HYP

# --- sft semi ---
run 1 $H_SHORT --cpu 4 --mem 32G scripts/run_eval_semi.sh $MH--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds -- $TASKS_HYP

# ============================================================
# IFEVAL (5 models × 99 tasks) — skipped for now (not effective in this setting)
# ============================================================

TASKS_IFE="ifeval-prompt_1 ifeval-prompt_2 ifeval-prompt_3 ifeval-prompt_4 ifeval-prompt_5 ifeval-prompt_6 ifeval-prompt_7 ifeval-prompt_8 ifeval-prompt_9 ifeval-prompt_10 ifeval-prompt_11 ifeval-prompt_12 ifeval-prompt_13 ifeval-prompt_15 ifeval-prompt_16 ifeval-prompt_17 ifeval-prompt_18 ifeval-prompt_19 ifeval-prompt_20 ifeval-prompt_21 ifeval-prompt_22 ifeval-prompt_23 ifeval-prompt_24 ifeval-prompt_25 ifeval-prompt_26 ifeval-prompt_27 ifeval-prompt_28 ifeval-prompt_29 ifeval-prompt_30 ifeval-prompt_32 ifeval-prompt_33 ifeval-prompt_34 ifeval-prompt_35 ifeval-prompt_36 ifeval-prompt_37 ifeval-prompt_38 ifeval-prompt_39 ifeval-prompt_40 ifeval-prompt_41 ifeval-prompt_42 ifeval-prompt_43 ifeval-prompt_44 ifeval-prompt_45 ifeval-prompt_46 ifeval-prompt_47 ifeval-prompt_48 ifeval-prompt_49 ifeval-prompt_50 ifeval-prompt_51 ifeval-prompt_52 ifeval-prompt_53 ifeval-prompt_54 ifeval-prompt_56 ifeval-prompt_57 ifeval-prompt_58 ifeval-prompt_59 ifeval-prompt_61 ifeval-prompt_63 ifeval-prompt_64 ifeval-prompt_65 ifeval-prompt_66 ifeval-prompt_67 ifeval-prompt_68 ifeval-prompt_70 ifeval-prompt_72 ifeval-prompt_73 ifeval-prompt_74 ifeval-prompt_75 ifeval-prompt_76 ifeval-prompt_77 ifeval-prompt_78 ifeval-prompt_79 ifeval-prompt_80 ifeval-prompt_82 ifeval-prompt_83 ifeval-prompt_84 ifeval-prompt_85 ifeval-prompt_87 ifeval-prompt_88 ifeval-prompt_89 ifeval-prompt_90 ifeval-prompt_91 ifeval-prompt_92 ifeval-prompt_93 ifeval-prompt_94 ifeval-prompt_95 ifeval-prompt_96 ifeval-prompt_97 ifeval-prompt_98 ifeval-prompt_99 ifeval-prompt_100 ifeval-prompt_102 ifeval-prompt_103 ifeval-prompt_104 ifeval-prompt_105 ifeval-prompt_106 ifeval-prompt_107 ifeval-prompt_108 ifeval-prompt_109"

MI=../models/v6-google--${MODEL}-delta0.15-epoch2--ifeval-concat-all--d2g--random--alpha1.0${TC_SUFFIX}

# --- pref-only labelonly (no vallogodds) ---
run 1 $H_IFE --cpu 4 --mem 32G scripts/run_eval_semi.sh $MI--full-completion--force-same-x--labelonly0.1$MSUF --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE

# --- pref-only labelonly (vallogodds) ---
run 1 $H_IFE --cpu 4 --mem 32G scripts/run_eval_semi.sh $MI--full-completion--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE

# --- comb labelonly ---
run 1 $H_IFE --cpu 4 --mem 32G scripts/run_eval_semi.sh $MI--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1$MSUF --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE

# --- comb semi ---
run 1 $H_IFE --cpu 4 --mem 32G scripts/run_eval_semi.sh $MI--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1$MSUF --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE

# --- sft semi ---
run 1 $H_IFE --cpu 4 --mem 32G scripts/run_eval_semi.sh $MI--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1$MSUF --neg-typcorr --log-odds --disc-shots-zero -- $TASKS_IFE

