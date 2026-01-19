CUDA_VISIBLE_DEVICES=1 python eval.py   --model  ../models/v5-google--gemma-2-2b-delta2.5-epoch2--hypernym--g2d--random--alpha1.0--ches-p0 --task hypernym   --use_full_completion_logprobs --variation kind-of --viz

CUDA_VISIBLE_DEVICES=1 python eval.py   --model  ../models/v5-google--gemma-2-2b-delta2.5-epoch2--hypernym--g2d--random--alpha1.0--ches-p25 --task hypernym   --use_full_completion_logprobs --variation kind-of --viz

CUDA_VISIBLE_DEVICES=1 python eval.py   --model  ../models/v5-google--gemma-2-2b-delta2.5-epoch2--hypernym--g2d--random--alpha1.0--ches-p50 --task hypernym   --use_full_completion_logprobs --variation kind-of --viz

CUDA_VISIBLE_DEVICES=1 python eval.py   --model  ../models/v5-google--gemma-2-2b-delta2.5-epoch2--hypernym--g2d--random--alpha1.0--ches-p75 --task hypernym   --use_full_completion_logprobs --variation kind-of --viz

CUDA_VISIBLE_DEVICES=1 python eval.py   --model  ../models/v5-google--gemma-2-2b-delta2.5-epoch2--hypernym--g2d--random--alpha1.0--ches-p100 --task hypernym   --use_full_completion_logprobs --variation kind-of --viz

