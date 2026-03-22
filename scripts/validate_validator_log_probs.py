#!/usr/bin/env python3
"""
Validate that validator (discriminator) probabilities are reasonable (p_yes + p_no ~= 1).
Sanity-checking that the model concentrates probability mass on Yes/No tokens.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent for imports
parent_dir = Path(__file__).resolve().parent.parent
src_path = parent_dir / "src"
sys.path.insert(0, str(src_path))

from transformers import AutoModelForCausalLM, AutoTokenizer

import tasks  # Triggers task registration
from task_registry import get_task, list_registered_tasks
from utils import get_final_logit_prob, get_L_prompt

YES_WORDS = ["Yes", " Yes", "YES", "yes", " yes"]
NO_WORDS = ["No", " No", "NO", "no", " no"]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_model(model_name, device, fp32=False):
    torch_dtype = torch.float32 if fp32 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def discover_tasks_by_regex(pattern):
    """Return task names matching the regex pattern."""
    all_tasks = list_registered_tasks()
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex '{pattern}': {e}")
    return [t for t in all_tasks if regex.search(t)]


def run_validation(
    model,
    tokenizer,
    task_name: str,
    device: str,
    max_samples,
    is_chat: bool,
    has_system_role: bool,
    split_name: str,
    LL: list,
    make_prompt,
    disc_shots: str = "zero",
):
    """
    Run validator on all (or max_samples) items for the given split.
    Returns (p_sums, n_skipped).
    """
    if not LL:
        return [], 0

    yestoks = [tokenizer.encode(w)[-1] for w in YES_WORDS]
    notoks = [tokenizer.encode(w)[-1] for w in NO_WORDS]

    n = len(LL) if max_samples is None else min(max_samples, len(LL))
    p_sums = []

    for i in tqdm(range(n), desc=f"{task_name} ({split_name})", leave=False):
        item = LL[i]
        prompt_disc = make_prompt(item, style="discriminator", shots=disc_shots).prompt
        probs_disc = get_final_logit_prob(
            prompt_disc,
            model,
            tokenizer,
            device,
            is_chat=is_chat,
            has_system_role=has_system_role,
        )
        p_yes = float(probs_disc[yestoks].sum().item())
        p_no = float(probs_disc[notoks].sum().item())
        p_sums.append(p_yes + p_no)

    return p_sums, len(LL) - n


def main():
    parser = argparse.ArgumentParser(
        description="Validate that validator probabilities sum to ~1"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="Model name or path",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        type=str,
        help="Single task name (e.g., ifeval-prompt_4, ifeval-concat)",
    )
    group.add_argument(
        "--task-regex",
        type=str,
        metavar="PATTERN",
        help="Regex to match task names (e.g., 'ifeval-prompt_.*')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task (for quick checks)",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use float32 for model (avoids bfloat16 quantization)",
    )
    parser.add_argument(
        "--save-hist",
        type=str,
        metavar="PATH",
        default=None,
        help="Save histogram to PNG",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Tolerance: flag p_yes+p_no outside [1-tol, 1+tol] as outliers (default: 0.05)",
    )
    parser.add_argument(
        "--disc-shots",
        type=str,
        default="zero",
        help="Discriminator shots: 'zero' or 'few' (default: zero)",
    )
    args = parser.parse_args()

    # Resolve task list
    if args.task:
        tasks_to_run = [args.task]
    else:
        tasks_to_run = discover_tasks_by_regex(args.task_regex)
        if not tasks_to_run:
            print(f"No tasks matched regex: {args.task_regex}")
            sys.exit(1)
        print(f"Matched {len(tasks_to_run)} tasks: {tasks_to_run[:5]}{'...' if len(tasks_to_run) > 5 else ''}")

    device = get_device()
    print(f"Loading model: {args.model} (device: {device})")
    model, tokenizer = init_model(args.model, device, fp32=args.fp32)

    is_chat = "instruct" in args.model.lower() or "-it" in args.model.lower()
    has_system_role = "llama" in args.model.lower()

    all_p_sums = []
    all_task_stats = []

    for task_name in tasks_to_run:
        task_config = get_task(task_name)
        if task_config is None and args.task_regex is None and args.task is not None:
            # Only validate known tasks when using --task
            print(f"  [SKIP] Unknown task: {task_name}")
            continue

        try:
            L_train, L_test, make_prompt = get_L_prompt(
                task_name, split_type="random", seed=0, sample_negative=False
            )
        except Exception as e:
            print(f"  [ERROR] {task_name}: {e}")
            continue

        task_p_sums = []
        task_n_skipped = 0
        for split_name, LL in [("train", L_train), ("test", L_test)]:
            if not LL:
                continue
            try:
                p_sums, n_skipped = run_validation(
                    model,
                    tokenizer,
                    task_name,
                    device,
                    args.max_samples,
                    is_chat,
                    has_system_role,
                    split_name,
                    LL,
                    make_prompt,
                    disc_shots=args.disc_shots,
                )
                task_p_sums.extend(p_sums)
                task_n_skipped += n_skipped
            except Exception as e:
                print(f"  [ERROR] {task_name}: {e}")
                continue

        if not task_p_sums:
            print(f"  [SKIP] {task_name}: no data")
            continue

        arr = np.array(task_p_sums)
        all_p_sums.extend(task_p_sums)
        stats = {
            "task": task_name,
            "n": len(task_p_sums),
            "skipped": task_n_skipped,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "pct_below": float(np.mean(arr < 1 - args.tol) * 100),
            "pct_above": float(np.mean(arr > 1 + args.tol) * 100),
        }
        all_task_stats.append(stats)

    # Per-task summary
    print("\n" + "=" * 80)
    print("Validator probability check: P(Yes) + P(No) should be ~1")
    print("=" * 80)

    for s in all_task_stats:
        status = "[OK]" if 1 - args.tol <= s["mean"] <= 1 + args.tol else "[WARN]"
        print(f"  {status} {s['task']}: n={s['n']} mean={s['mean']:.4f} std={s['std']:.4f} "
              f"min={s['min']:.4f} max={s['max']:.4f} "
              f"(outliers: {s['pct_below']:.1f}% below, {s['pct_above']:.1f}% above)")

    if not all_p_sums:
        print("No data processed.")
        sys.exit(1)

    # Overall
    arr = np.array(all_p_sums)
    print("\n" + "-" * 40)
    print(f"Overall: n={len(arr)} mean={np.mean(arr):.4f} std={np.std(arr):.4f} "
          f"min={np.min(arr):.4f} max={np.max(arr):.4f}")
    n_out = np.sum((arr < 1 - args.tol) | (arr > 1 + args.tol))
    print(f"Outliers (outside [1±{args.tol}]): {n_out} ({100 * n_out / len(arr):.1f}%)")

    if args.save_hist:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.hist(arr, bins=50, edgecolor="black", alpha=0.7)
        plt.axvline(1, color="green", linestyle="--", label="target=1")
        plt.axvline(1 - args.tol, color="orange", linestyle=":", alpha=0.7)
        plt.axvline(1 + args.tol, color="orange", linestyle=":", alpha=0.7)
        plt.xlabel("P(Yes) + P(No)")
        plt.ylabel("Count")
        plt.title(f"Validator probability sum (n={len(arr)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_hist)
        print(f"\nHistogram saved to {args.save_hist}")


if __name__ == "__main__":
    main()
