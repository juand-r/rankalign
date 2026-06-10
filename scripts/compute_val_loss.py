#!/usr/bin/env python3
"""
Compute validation (test-set) loss for checkpoints at base, epoch 0, 1, 2.

Computes the same loss components as training (preference, NLL-G, NLL-V)
on held-out test data, to diagnose overfitting.

Usage:
    python scripts/compute_val_loss.py \
        --model google/gemma-2-9b-it \
        --task membership-sans-rosch-v0 \
        --setting s4 \
        --models-dir /datastor2/jdr/rankalign/models2-rerun-wandb \
        --output analysis/tables/val_loss.csv

The script:
1. Loads the task's test data (L_test from the task registry)
2. Forms all positive-vs-negative pairs
3. For each checkpoint (base, ep0, ep1, ep2):
   - Loads the model
   - Computes gen scores (sum log P(completion|prompt)) for each item
   - Computes val scores (log-odds of "Yes" given disc prompt) for each item
   - Derives: preference_loss, nll_generator_loss, nll_validator_loss
4. Saves results to CSV
"""

import argparse
import sys
import os
import json
import csv
from pathlib import Path
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_task, make_and_format_data
import tasks  # Triggers task registration (ifeval-concat, membership-sans-rosch-v0, etc.)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_completion_logprobs(prompt, completion, model, tokenizer, device):
    """Compute sum of log-probs for completion tokens given prompt."""
    full_text = prompt + completion
    enc_full = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=2048).to(device)
    enc_prompt = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    prompt_len = enc_prompt['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model(**enc_full)
        logits = outputs.logits  # (1, seq_len, vocab)

    log_probs = torch.log_softmax(logits[0], dim=-1)
    input_ids = enc_full['input_ids'][0]

    # Sum log-probs of completion tokens (shifted by 1 for autoregressive)
    total = 0.0
    n_tokens = 0
    for i in range(prompt_len, len(input_ids)):
        if i > 0:
            total += log_probs[i - 1, input_ids[i]].item()
            n_tokens += 1

    return total, n_tokens


def get_val_logodds(prompt, model, tokenizer, device, yes_token_id, no_token_id):
    """Compute log-odds of 'Yes' vs 'No' for a discriminator prompt."""
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[0, -1, :]  # last position

    log_probs = torch.log_softmax(logits, dim=-1)
    logodds = log_probs[yes_token_id] - log_probs[no_token_id]
    return logodds.item()


def find_checkpoints(model_name, task, setting, models_dir):
    """Find epoch 0, 1, 2 checkpoint directories for a given setting."""
    from scripts_settings import get_setting_suffix
    suffix = get_setting_suffix(setting)

    model_short = model_name.replace('/', '--')
    models_path = Path(models_dir)

    checkpoints = {}
    for epoch in [0, 1, 2]:
        pattern = f"v7-{model_short}-delta*-epoch{epoch}--{task}-all--d2g--random--alpha1.0{suffix}_merged"
        matches = list(models_path.glob(pattern))
        if matches:
            checkpoints[epoch] = str(sorted(matches)[-1])  # most recent
        else:
            # Try without _merged
            pattern2 = f"v7-{model_short}-delta*-epoch{epoch}--{task}-all--d2g--random--alpha1.0{suffix}"
            matches2 = [m for m in models_path.glob(pattern2) if '_merged' not in str(m)]
            if matches2:
                checkpoints[epoch] = str(sorted(matches2)[-1])

    return checkpoints


def _load_test_data(test_task_name):
    """Return (L_test, task_config_for_prompts) for the given test-task name.

    Special case: 'rosch-all' aggregates L_test across every registered rosch-*
    task (which is how membership models are evaluated for held-out
    generalization, since membership-sans-rosch-v0's load_data returns no
    test items).
    """
    if test_task_name == "rosch-all":
        from task_registry import list_registered_tasks
        rosch_names = sorted(name for name in list_registered_tasks() if name.startswith("rosch-"))
        if not rosch_names:
            raise ValueError("No rosch-* tasks registered")
        L_test_all = []
        rosch_cfg = None
        for name in rosch_names:
            cfg = get_task(name)
            if cfg is None:
                continue
            _, items = cfg['load_data'](seed=0, split_type='random')
            L_test_all.extend(items)
            if rosch_cfg is None:
                rosch_cfg = cfg
        print(f"  Aggregated rosch-all from {len(rosch_names)} per-category tasks: "
              f"{len(L_test_all)} total test items")
        return L_test_all, rosch_cfg

    cfg = get_task(test_task_name)
    if cfg is None:
        raise ValueError(f"Task {test_task_name} not found in registry")
    _, L_test = cfg['load_data'](seed=0, split_type='random')
    return L_test, cfg


def main():
    parser = argparse.ArgumentParser(description="Compute validation loss on test set")
    parser.add_argument("--model", required=True, help="Base model name (e.g. google/gemma-2-9b-it)")
    parser.add_argument("--task", required=True, help="Training task name (used to find checkpoints, e.g. membership-sans-rosch-v0)")
    parser.add_argument("--test-task", default=None,
                        help="Task to load test data from. Defaults to --task. Use 'rosch-all' to aggregate "
                             "all rosch-* test sets (the held-out set for membership models).")
    parser.add_argument("--setting", required=True, help="Setting (s1-s7)")
    parser.add_argument("--models-dir", required=True, help="Directory with trained checkpoints")
    parser.add_argument("--output", default="analysis/tables/val_loss.csv", help="Output CSV path")
    parser.add_argument("--max-pairs", type=int, default=500, help="Max pairs to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve training task (used elsewhere only for documentation in CSV output)
    train_task_config = get_task(args.task)
    if train_task_config is None:
        raise ValueError(f"Training task {args.task} not found in registry")

    # Resolve test task (where we get held-out items + the prompts used at eval).
    test_task_name = args.test_task or args.task
    print(f"Training task: {args.task}  |  Test task: {test_task_name}")
    L_test, task_config = _load_test_data(test_task_name)
    print(f"Test set: {len(L_test)} items")

    # Split test items into positive/negative
    positives = [item for item in L_test if task_config['get_label'](item) == 'yes']
    negatives = [item for item in L_test if task_config['get_label'](item) == 'no']
    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    if not positives or not negatives:
        print("ERROR: Need both positive and negative items in test set!")
        return

    # Form pairs (positive, negative) for preference loss
    rng = np.random.default_rng(args.seed)
    all_pairs = list(product(range(len(positives)), range(len(negatives))))
    if len(all_pairs) > args.max_pairs:
        pair_idx = rng.choice(len(all_pairs), size=args.max_pairs, replace=False)
        pairs = [all_pairs[i] for i in pair_idx]
    else:
        pairs = all_pairs
    print(f"  Pairs for eval: {len(pairs)}")

    # Determine which checkpoints to evaluate
    # First: base model, then epoch 0, 1, 2
    checkpoints = {"base": args.model}

    # Find trained checkpoints
    ckpt_paths = find_checkpoints(args.model, args.task, args.setting, args.models_dir)
    for ep, path in ckpt_paths.items():
        checkpoints[f"epoch{ep}"] = path
    print(f"\nCheckpoints found: {list(checkpoints.keys())}")
    for k, v in checkpoints.items():
        print(f"  {k}: {v}")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find Yes/No token IDs for validator
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[-1]
    print(f"\nYes token id: {yes_id}, No token id: {no_id}")

    # Prepare prompts for all test items
    gen_prompts_pos = []
    gen_prompts_neg = []
    disc_prompts_pos = []
    disc_prompts_neg = []
    labels_pos = []
    labels_neg = []

    for item in positives:
        pc = task_config['make_prompt'](item, style='generator', shots='zero')
        gen_prompts_pos.append((pc.prompt, pc.completion))
        dc = task_config['make_prompt'](item, style='discriminator', shots='zero')
        disc_prompts_pos.append(dc.prompt)
        labels_pos.append(1)

    for item in negatives:
        pc = task_config['make_prompt'](item, style='generator', shots='zero')
        gen_prompts_neg.append((pc.prompt, pc.completion))
        dc = task_config['make_prompt'](item, style='discriminator', shots='zero')
        disc_prompts_neg.append(dc.prompt)
        labels_neg.append(0)

    results = []

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name} ({ckpt_path})")
        print(f"{'='*60}")

        # Load model
        if ckpt_name == "base":
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True
            )
        else:
            # Merged checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True
            )
        model.eval()

        # Compute gen scores for all items
        print("  Computing gen scores (positives)...")
        gen_scores_pos = []
        for prompt, completion in tqdm(gen_prompts_pos, desc="gen_pos"):
            score, _ = get_completion_logprobs(prompt, completion, model, tokenizer, device)
            gen_scores_pos.append(score)

        print("  Computing gen scores (negatives)...")
        gen_scores_neg = []
        for prompt, completion in tqdm(gen_prompts_neg, desc="gen_neg"):
            score, _ = get_completion_logprobs(prompt, completion, model, tokenizer, device)
            gen_scores_neg.append(score)

        # Compute val scores (log-odds) for all items
        print("  Computing val scores (positives)...")
        val_scores_pos = []
        for dp in tqdm(disc_prompts_pos, desc="val_pos"):
            score = get_val_logodds(dp, model, tokenizer, device, yes_id, no_id)
            val_scores_pos.append(score)

        print("  Computing val scores (negatives)...")
        val_scores_neg = []
        for dp in tqdm(disc_prompts_neg, desc="val_neg"):
            score = get_val_logodds(dp, model, tokenizer, device, yes_id, no_id)
            val_scores_neg.append(score)

        # Compute losses
        # 1. Preference loss: for each (pos, neg) pair, -log(sigmoid(gen_pos - gen_neg))
        pref_losses = []
        for pi, ni in pairs:
            diff = gen_scores_pos[pi] - gen_scores_neg[ni]
            pref_loss = -np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-12)
            pref_losses.append(pref_loss)
        mean_pref_loss = np.mean(pref_losses)

        # 2. NLL-G: -mean(gen_score) for positive items only
        mean_nll_g = -np.mean(gen_scores_pos)

        # 3. NLL-V: BCE loss for validator predictions
        # For positives: -log(sigmoid(val_score))
        # For negatives: -log(sigmoid(-val_score))
        bce_losses = []
        for vs in val_scores_pos:
            bce_losses.append(-np.log(1.0 / (1.0 + np.exp(-vs)) + 1e-12))
        for vs in val_scores_neg:
            bce_losses.append(-np.log(1.0 / (1.0 + np.exp(vs)) + 1e-12))
        mean_nll_v = np.mean(bce_losses)

        row = {
            "checkpoint": ckpt_name,
            "setting": args.setting,
            "model": args.model,
            "task": args.task,
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "n_pairs": len(pairs),
            "preference_loss": mean_pref_loss,
            "nll_generator_loss": mean_nll_g,
            "nll_validator_loss": mean_nll_v,
            "total_loss": mean_pref_loss + mean_nll_g + mean_nll_v,
            "mean_gen_score_pos": np.mean(gen_scores_pos),
            "mean_gen_score_neg": np.mean(gen_scores_neg),
            "mean_val_score_pos": np.mean(val_scores_pos),
            "mean_val_score_neg": np.mean(val_scores_neg),
        }
        results.append(row)
        print(f"  pref_loss={mean_pref_loss:.4f}  nll_g={mean_nll_g:.4f}  nll_v={mean_nll_v:.4f}  total={row['total_loss']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append if file exists
    file_exists = output_path.exists()
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    print(f"{'Checkpoint':<12} {'Pref Loss':>10} {'NLL-G':>10} {'NLL-V':>10} {'Total':>10}")
    print("-" * 56)
    for r in results:
        print(f"{r['checkpoint']:<12} {r['preference_loss']:>10.4f} {r['nll_generator_loss']:>10.4f} {r['nll_validator_loss']:>10.4f} {r['total_loss']:>10.4f}")


if __name__ == "__main__":
    main()
