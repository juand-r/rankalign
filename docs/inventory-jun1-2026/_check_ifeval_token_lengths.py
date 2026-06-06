#!/usr/bin/env python3
"""Run ON mll (gemma4 venv). Tokenize ifeval-concat TRAIN sequences with the
Qwen3.5 tokenizer and report the length distribution vs --max-seq-len 1024.

Measures the generator training sequence the trainer builds:
   chat_template(make_prompt(generator, zero), enable_thinking=False) + completion
and the discriminator sequence (prompt + ' yes'/' no'). The generator is the long one.
"""
import os
import sys

REPO = "/datastor2/jdr/rankalign"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "tasks"))

import numpy as np
import ifeval_concat  # registers + provides load_data / make_prompt / get_completion / get_label
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
L_train, L_test = ifeval_concat.load_data()
print(f"ifeval-concat: {len(L_train)} train items, {len(L_test)} test items")


def chat_len(prompt_text):
    return len(tok(prompt_text, add_special_tokens=False)["input_ids"])  # raw content tokens (+~30 chat overhead)

def _unused(prompt_text):
    kw = dict(tokenize=True, add_generation_prompt=True)
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                      enable_thinking=False, **kw)
    except TypeError:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt_text}], **kw)
    return len(ids)


gen_lens, disc_lens = [], []
for item in L_train:
    gp = ifeval_concat.make_prompt(item, style="generator", shots="zero")
    comp = ifeval_concat.get_completion(item)
    comp_ids = tok(comp, add_special_tokens=False)["input_ids"]
    gen_lens.append(chat_len(gp) + len(comp_ids))
    dp = ifeval_concat.make_prompt(item, style="discriminator", shots="zero")
    disc_lens.append(chat_len(dp) + 2)  # + ' yes'/' no'

for name, arr in [("GENERATOR (prompt+completion)", np.array(gen_lens)),
                  ("DISCRIMINATOR (prompt+label)", np.array(disc_lens))]:
    print(f"\n=== {name} — token length over {len(arr)} train items ===")
    for q in (50, 90, 95, 99):
        print(f"   p{q:>2}: {np.percentile(arr, q):7.0f}")
    print(f"   max: {arr.max():7d}   mean: {arr.mean():7.1f}")
    for thr in (1024, 1536, 2048):
        n = int((arr > thr).sum())
        print(f"   > {thr}: {n:5d}  ({100*n/len(arr):5.1f}%)")
