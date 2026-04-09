echo "Do not run this script."
exit 1

cd /datastor2/jocelyn/rankalign/scripts

# # Gemma-2-9B-IT | IFEval | Currently Running ...
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant base --semi-mode none

# # Plain | Label-only
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant sft       --semi-mode semi --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 2 48 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --tc --variant all_terms --semi-mode semi --semi-ratio 0.1

# Gemma-2-9B-IT | PlausibleQA | N/A
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant base --semi-mode none

# # Plain | Label-only
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --tc --variant all_terms --semi-mode semi --semi-ratio 0.1

# # Gemma-2-9B-IT | AmbigQA | N/A
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant base --semi-mode none

# # Plain | Label-only
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --tc --variant all_terms --semi-mode semi --semi-ratio 0.1

# # Gemma-2-9B-IT | Hypernym Concat | N/A
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant base --semi-mode none

# # Plain | Label-only
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 24 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant all_terms --semi-mode semi --semi-ratio 0.1

# # Gemma-2-2B-IT | PlausibleQA | N/A
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant base --semi-mode none

# # Plain | Label-only
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_plausibleqa.sh 1 --model google/gemma-2-2b-it --task plausibleqa --tc --variant all_terms --semi-mode semi --semi-ratio 0.1

# # Gemma-2-2B-IT | AmbigQA | N/A
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant base --semi-mode none

# # Plain | Label-only
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --tc --variant all_terms --semi-mode semi --semi-ratio 0.1

# # Gemma-2-2B-IT | Hypernym Concat | N/A
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant base --semi-mode none

# # Plain | Label-only
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # Plain | Semisupervised
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode semi --semi-ratio 0.1

# # TC-self | Label-only
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant pref_only --semi-mode labelonly --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant all_terms --semi-mode labelonly --semi-ratio 0.1

# # TC-self | Semisupervised
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant pref_only --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant sft       --semi-mode semi --semi-ratio 0.1
# run 1 10 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --tc --variant all_terms --semi-mode semi --semi-ratio 0.1