echo "Do not run this script."
exit 1

cd /datastor2/jocelyn/rankalign/scripts

# # Gemma-2-9B-IT | IFEval 
# run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant base --semi-mode none --eval-only

# # Plain | Label-only
# run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant pref_only --semi-mode labelonly --semi-ratio 0.1 --eval-only
# run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant sft       --semi-mode labelonly --semi-ratio 0.1 --eval-only
# run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant all_terms --semi-mode labelonly --semi-ratio 0.1 --eval-only

# Plain | Semisupervised
run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant pref_only --semi-mode semi --semi-ratio 0.1 --eval-only
run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant sft       --semi-mode semi --semi-ratio 0.1 --eval-only
run 2 8 ./run_ifeval_concat.sh 2 --model google/gemma-2-9b-it --task ifeval-concat --variant all_terms --semi-mode semi --semi-ratio 0.1 --eval-only

# Gemma-2-9B-IT | PlausibleQA | N/A
run 1 5 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant base --semi-mode none --eval-only

# Plain | Label-only
run 1 4 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant pref_only --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 4 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant sft       --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 4 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant all_terms --semi-mode labelonly --semi-ratio 0.1 --eval-only

# Plain | Semisupervised
run 1 4 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant pref_only --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 4 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant sft       --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 4 ./run_plausibleqa.sh 1 --model google/gemma-2-9b-it --task plausibleqa --variant all_terms --semi-mode semi --semi-ratio 0.1 --eval-only

# Gemma-2-9B-IT | AmbigQA | N/A
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant base --semi-mode none --eval-only

# Plain | Label-only
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant pref_only --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant sft       --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant all_terms --semi-mode labelonly --semi-ratio 0.1 --eval-only

# Plain | Semisupervised
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant pref_only --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant sft       --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 4 ./run_ambigqa.sh 1 --model google/gemma-2-9b-it --task ambigqa --variant all_terms --semi-mode semi --semi-ratio 0.1 --eval-only

# Gemma-2-9B-IT | Hypernym Concat | N/A
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant base --semi-mode none --eval-only

# Plain | Label-only
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode labelonly --semi-ratio 0.1 --eval-only

# Plain | Semisupervised
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 4 ./run_hypernym_concat.sh 1 --model google/gemma-2-9b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode semi --semi-ratio 0.1 --eval-only




####### 2b models #######

# Plain | Semisupervised
run 1 2 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant pref_only --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 2 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant sft       --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 2 ./run_ambigqa.sh 1 --model google/gemma-2-2b-it --task ambigqa --variant all_terms --semi-mode semi --semi-ratio 0.1 --eval-only

# Gemma-2-2B-IT | Hypernym Concat | N/A
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant base --semi-mode none --eval-only

# Plain | Label-only
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode labelonly --semi-ratio 0.1 --eval-only
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode labelonly --semi-ratio 0.1 --eval-only

# Plain | Semisupervised
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant pref_only --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant sft       --semi-mode semi --semi-ratio 0.1 --eval-only
run 1 2 ./run_hypernym_concat.sh 1 --model google/gemma-2-2b-it --task hypernym-concat-bananas-to-dogs-double --variant all_terms --semi-mode semi --semi-ratio 0.1 --eval-only