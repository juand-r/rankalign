#!/usr/bin/env python3
import os
import torch

os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/workspace/.cache/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/hub'
os.environ['HF_HUB_DISABLE_XET'] = '1'

ADAPTER = ('/workspace/models_g4it/'
           'v6-google--gemma-4-31B-it-delta0.15-epoch0'
           '--humaneval-v2.1correct-upper-all--d2g--random'
           '--alpha1.0--full-completion--semi0.1')
MERGED = ADAPTER + '_merged'

print(f'Adapter: {ADAPTER}')
print(f'Merged output: {MERGED}')

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

print('Loading LoRA model (float16, device_map=auto)...')
model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER,
    torch_dtype=torch.float16,
    device_map='auto',
    low_cpu_mem_usage=True,
)
print('Merging LoRA into base...')
merged = model.merge_and_unload()
print(f'Saving merged model to {MERGED} ...')
merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size='2GB')
print('Saving tokenizer...')
tok = AutoTokenizer.from_pretrained(ADAPTER)
tok.save_pretrained(MERGED)
print('DONE: merge complete.')
