#!/usr/bin/env python3
"""
Compute held-out (test-set) loss for HumanEval checkpoints at base, epoch 0, 1, 2.

HumanEval variant of compute_val_loss.py. Copied (NOT edited in place) so the original
IFEval/membership val-loss path stays untouched. Differences from the original:

1. Checkpoint paths are passed explicitly via --ep0-path/--ep1-path/--ep2-path (computed
   by the sbatch's proven ckpt_path() glob), so we do NOT rely on the IFEval-specific
   find_checkpoints() / scripts_settings suffix map (which is stale for HumanEval, e.g. it
   omits `vallogodds` from s1). find_checkpoints() is kept only for backward compatibility.
2. Adapter-aware model loading: gemma-4 HumanEval checkpoints are LoRA adapters (load base
   + PeftModel); qwen-3.5 checkpoints are merged full models (load directly). Detected by
   the presence of adapter_config.json in the checkpoint dir.
3. The held-out test set is the union of the 82 per-problem TEST tasks
   (humaneval-v2.1correct-{ds}-<slug>), because the base task's load_data is train-only and
   returns empty L_test. Selected via --test-task humaneval-{upper,multi}-all. Mirrors the
   original's `rosch-all` aggregation for membership models.
4. Incremental, resumable CSV: each checkpoint's row is appended as soon as it is computed,
   and already-computed (model, task, setting, checkpoint) rows are skipped on restart, so a
   4-checkpoint job survives a timeout.

Loss components (identical to the original): preference, NLL-G, NLL-V on held-out data.

Usage (the sbatch passes everything; manual example):
    python scripts/compute_val_loss_he.py \
        --model google/gemma-4-31B-it \
        --task humaneval-v2.1correct-upper \
        --test-task humaneval-upper-all \
        --setting s4 \
        --ep0-path <dir>/...epoch0... --ep1-path ... --ep2-path ... \
        --output analysis/tables/he_val_loss.csv
"""

# Max sequence length for scoring. Bumped from the original 2048 to 4096 because HumanEval
# candidates are full programs; a per-checkpoint warning fires if anything actually hits it.
MAX_LEN = 4096
_TRUNC_COUNT = 0

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
    global _TRUNC_COUNT
    full_text = prompt + completion
    enc_full = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(device)
    enc_prompt = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN
    )
    prompt_len = enc_prompt["input_ids"].shape[1]
    if enc_full["input_ids"].shape[1] >= MAX_LEN:
        _TRUNC_COUNT += 1

    with torch.no_grad():
        outputs = model(**enc_full)
        logits = outputs.logits  # (1, seq_len, vocab)

    log_probs = torch.log_softmax(logits[0], dim=-1)
    input_ids = enc_full["input_ids"][0]

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
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(device)
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

    model_short = model_name.replace("/", "--")
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
            matches2 = [
                m for m in models_path.glob(pattern2) if "_merged" not in str(m)
            ]
            if matches2:
                checkpoints[epoch] = str(sorted(matches2)[-1])

    return checkpoints


def load_model(ckpt_path, base_model, dtype=torch.bfloat16):
    """Load a checkpoint, auto-detecting LoRA adapter vs merged/full model.

    gemma-4 HumanEval checkpoints are LoRA adapters (adapter_config.json present) -> load
    the base model and wrap with PeftModel. qwen-3.5 checkpoints are merged full models ->
    load directly. The "base" pseudo-checkpoint passes a HF model id, which has no
    adapter_config.json on disk and so loads directly.
    """
    is_adapter = (Path(ckpt_path) / "adapter_config.json").exists()
    if is_adapter:
        print(f"  [load] LoRA adapter detected -> base={base_model} + adapter")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        return PeftModel.from_pretrained(base, ckpt_path)
    print("  [load] merged/full model")
    return AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )


def _aggregate_perproblem_test(ds):
    """Aggregate L_test across the 82 per-problem TEST tasks for a HumanEval dataset.

    The base task humaneval-v2.1correct-{ds} is train-only (empty L_test); the held-out test
    set lives in the per-problem tasks humaneval-v2.1correct-{ds}-<slug>. Excludes the
    per-problem TRAIN tasks (humaneval-v2.1correct-{ds}-train-<slug>).
    """
    from task_registry import list_registered_tasks

    prefix = f"humaneval-v2.1correct-{ds}-"
    names = sorted(
        n
        for n in list_registered_tasks()
        if n.startswith(prefix) and "-train-" not in n
    )
    if not names:
        raise ValueError(
            f"No per-problem test tasks registered for ds={ds} (prefix {prefix})"
        )
    L_test_all = []
    cfg0 = None
    for name in names:
        cfg = get_task(name)
        if cfg is None:
            continue
        _, items = cfg["load_data"](seed=0, split_type="random")
        L_test_all.extend(items)
        if cfg0 is None:
            cfg0 = cfg
    print(
        f"  Aggregated humaneval-{ds}-all from {len(names)} per-problem test tasks: "
        f"{len(L_test_all)} total test items"
    )
    return L_test_all, cfg0


def _load_test_data(test_task_name):
    """Return (L_test, task_config_for_prompts) for the given test-task name.

    Special cases:
      - 'rosch-all' aggregates L_test across every registered rosch-* task (membership
        held-out set; membership-sans-rosch-v0's load_data returns no test items).
      - 'humaneval-{upper,multi}-all' aggregates L_test across the 82 per-problem HumanEval
        test tasks (the base humaneval-v2.1correct-{ds} task is train-only).
    """
    if test_task_name in ("humaneval-upper-all", "humaneval-multi-all"):
        ds = "upper" if "upper" in test_task_name else "multi"
        return _aggregate_perproblem_test(ds)

    if test_task_name == "rosch-all":
        from task_registry import list_registered_tasks

        rosch_names = sorted(
            name for name in list_registered_tasks() if name.startswith("rosch-")
        )
        if not rosch_names:
            raise ValueError("No rosch-* tasks registered")
        L_test_all = []
        rosch_cfg = None
        for name in rosch_names:
            cfg = get_task(name)
            if cfg is None:
                continue
            _, items = cfg["load_data"](seed=0, split_type="random")
            L_test_all.extend(items)
            if rosch_cfg is None:
                rosch_cfg = cfg
        print(
            f"  Aggregated rosch-all from {len(rosch_names)} per-category tasks: "
            f"{len(L_test_all)} total test items"
        )
        return L_test_all, rosch_cfg

    cfg = get_task(test_task_name)
    if cfg is None:
        raise ValueError(f"Task {test_task_name} not found in registry")
    _, L_test = cfg["load_data"](seed=0, split_type="random")
    return L_test, cfg


def main():
    parser = argparse.ArgumentParser(description="Compute validation loss on test set")
    parser.add_argument(
        "--model", required=True, help="Base model name (e.g. google/gemma-2-9b-it)"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Training task name (used to find checkpoints, e.g. membership-sans-rosch-v0)",
    )
    parser.add_argument(
        "--test-task",
        default=None,
        help="Task to load test data from. Defaults to --task. Use 'rosch-all' to aggregate "
        "all rosch-* test sets (the held-out set for membership models).",
    )
    parser.add_argument("--setting", required=True, help="Setting (s1-s7, s13)")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory with trained checkpoints (only used by the legacy "
        "find_checkpoints() path; not needed when --epN-path are given)",
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Base model id/path for the 'base' checkpoint AND for loading LoRA "
        "adapters. Defaults to --model.",
    )
    parser.add_argument(
        "--ep0-path", default=None, help="Explicit epoch-0 checkpoint dir"
    )
    parser.add_argument(
        "--ep1-path", default=None, help="Explicit epoch-1 checkpoint dir"
    )
    parser.add_argument(
        "--ep2-path", default=None, help="Explicit epoch-2 checkpoint dir"
    )
    parser.add_argument(
        "--output", default="analysis/tables/he_val_loss.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=500,
        help="Max (pos,neg) pairs for preference loss",
    )
    parser.add_argument(
        "--max-items-per-class",
        type=int,
        default=None,
        help="Cap positives/negatives scored per class (default: all). Use only "
        "if a job is too slow, and note it — changes the held-out set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for pair/item sampling"
    )
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
    positives = [item for item in L_test if task_config["get_label"](item) == "yes"]
    negatives = [item for item in L_test if task_config["get_label"](item) == "no"]
    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    if not positives or not negatives:
        print("ERROR: Need both positive and negative items in test set!")
        return

    rng = np.random.default_rng(args.seed)

    # Optional per-class cap (off by default = use the full held-out set, mirroring original)
    if args.max_items_per_class is not None:
        k = args.max_items_per_class
        if len(positives) > k:
            positives = [
                positives[i] for i in rng.choice(len(positives), size=k, replace=False)
            ]
        if len(negatives) > k:
            negatives = [
                negatives[i] for i in rng.choice(len(negatives), size=k, replace=False)
            ]
        print(
            f"  Capped to {len(positives)} pos / {len(negatives)} neg (--max-items-per-class={k})"
        )

    # Form pairs (positive, negative) for preference loss
    all_pairs = list(product(range(len(positives)), range(len(negatives))))
    if len(all_pairs) > args.max_pairs:
        pair_idx = rng.choice(len(all_pairs), size=args.max_pairs, replace=False)
        pairs = [all_pairs[i] for i in pair_idx]
    else:
        pairs = all_pairs
    print(f"  Pairs for eval: {len(pairs)}")

    # Determine which checkpoints to evaluate: base, then epoch 0, 1, 2.
    base_model = args.base_model_path or args.model
    checkpoints = {"base": base_model}

    explicit = {0: args.ep0_path, 1: args.ep1_path, 2: args.ep2_path}
    if any(explicit.values()):
        # Preferred path: paths supplied by the sbatch's proven ckpt_path() glob.
        for ep, path in explicit.items():
            if path:
                checkpoints[f"epoch{ep}"] = path
    else:
        # Legacy fallback (IFEval/membership naming only).
        if not args.models_dir:
            raise ValueError(
                "Provide --ep0/1/2-path (HumanEval) or --models-dir (legacy)."
            )
        ckpt_paths = find_checkpoints(
            args.model, args.task, args.setting, args.models_dir
        )
        for ep, path in ckpt_paths.items():
            checkpoints[f"epoch{ep}"] = path

    print(f"\nCheckpoints to evaluate: {list(checkpoints.keys())}")
    for k, v in checkpoints.items():
        print(f"  {k}: {v}")

    # Resume: skip checkpoints already present in the output CSV for this (model, task, setting).
    done = set()
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                if (
                    r.get("model") == base_model
                    and r.get("task") == args.task
                    and r.get("setting") == args.setting
                ):
                    done.add(r.get("checkpoint"))
        if done:
            print(f"  Resume: already done {sorted(done)} -> skipping those")

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
        pc = task_config["make_prompt"](item, style="generator", shots="zero")
        gen_prompts_pos.append((pc.prompt, pc.completion))
        dc = task_config["make_prompt"](item, style="discriminator", shots="zero")
        disc_prompts_pos.append(dc.prompt)
        labels_pos.append(1)

    for item in negatives:
        pc = task_config["make_prompt"](item, style="generator", shots="zero")
        gen_prompts_neg.append((pc.prompt, pc.completion))
        dc = task_config["make_prompt"](item, style="discriminator", shots="zero")
        disc_prompts_neg.append(dc.prompt)
        labels_neg.append(0)

    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def append_row(row):
        """Append one checkpoint's row immediately (resumable on timeout)."""
        write_header = not output_path.exists()
        with open(output_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    for ckpt_name, ckpt_path in checkpoints.items():
        if ckpt_name in done:
            print(f"\n[skip] {ckpt_name} already in CSV")
            continue
        global _TRUNC_COUNT
        _TRUNC_COUNT = 0
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ckpt_name} ({ckpt_path})")
        print(f"{'=' * 60}")

        # Load model (adapter-aware: gemma-4 = LoRA on base, qwen = merged full model)
        model = load_model(ckpt_path, base_model)
        model.eval()

        # Compute gen scores for all items
        print("  Computing gen scores (positives)...")
        gen_scores_pos = []
        for prompt, completion in tqdm(gen_prompts_pos, desc="gen_pos"):
            score, _ = get_completion_logprobs(
                prompt, completion, model, tokenizer, device
            )
            gen_scores_pos.append(score)

        print("  Computing gen scores (negatives)...")
        gen_scores_neg = []
        for prompt, completion in tqdm(gen_prompts_neg, desc="gen_neg"):
            score, _ = get_completion_logprobs(
                prompt, completion, model, tokenizer, device
            )
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
            "model": base_model,
            "task": args.task,
            "test_task": test_task_name,
            "ckpt_path": ckpt_path,
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
        if _TRUNC_COUNT:
            print(
                f"  WARNING: {_TRUNC_COUNT} sequences hit MAX_LEN={MAX_LEN} (truncated)."
            )
        print(
            f"  pref_loss={mean_pref_loss:.4f}  nll_g={mean_nll_g:.4f}  nll_v={mean_nll_v:.4f}  total={row['total_loss']:.4f}"
        )
        append_row(row)  # incremental write -> resumable on timeout
        print(f"  [saved] {ckpt_name} -> {output_path}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    if not results:
        print("\nNothing computed (all checkpoints already done).")
        return

    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    print(
        f"{'Checkpoint':<12} {'Pref Loss':>10} {'NLL-G':>10} {'NLL-V':>10} {'Total':>10}"
    )
    print("-" * 56)
    for r in results:
        print(
            f"{r['checkpoint']:<12} {r['preference_loss']:>10.4f} {r['nll_generator_loss']:>10.4f} {r['nll_validator_loss']:>10.4f} {r['total_loss']:>10.4f}"
        )


if __name__ == "__main__":
    main()
