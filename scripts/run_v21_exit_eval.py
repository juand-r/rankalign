"""Score all v2.1 test items with logit-lens exit layers in one forward pass.

Loads gemma-4-31b-it ONCE, iterates over all 82 per-task test CSVs, and
scores each (prompt, completion) pair using the full model + exit layers
[0.25, 0.5, 0.75] simultaneously (output_hidden_states=True, single pass).

Output: JSONL with one row per test item containing:
  task_id, row_idx, correct, model, strategy, temperature, num_tokens,
  logp_full, logp_exit025, logp_exit050, logp_exit075

Usage:
  python run_v21_exit_eval.py \
      --model google/gemma-4-31B-it \
      --v21-dir /workspace/rankalign/data/humaneval/v2.1 \
      --output /workspace/v21_exit_scores.jsonl \
      [--exit-layers 0.25 0.5 0.75] \
      [--resume]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
RANKALIGN_SRC = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(RANKALIGN_SRC))

from utils import _get_model_lm_components, get_model_input_device
from tasks.humaneval import make_prompt_v2
import tasks  # trigger task registration (needed for make_prompt_v2 imports)


def load_v21_csv(path: Path) -> list[dict]:
    """Load one per-task v2.1 CSV, returning rows with question/answer/correct/etc."""
    fields = ("question", "answer", "correct", "strategy", "model", "temperature", "task_id", "error")
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            item = {k: row.get(k, "") for k in fields}
            item["_row_idx"] = i
            if item["answer"] and item["correct"]:
                rows.append(item)
    return rows


def build_prompt_and_completion(item: dict) -> tuple[str, str]:
    """Build the v2 format-C generator prompt and completion using the exact make_prompt_v2 logic."""
    pc = make_prompt_v2(item, style="generator", shots="zero")
    return pc.prompt, pc.completion


def score_item(
    prompt: str,
    completion: str,
    model,
    tokenizer,
    exit_fractions: list[float],
    norm,
    lm_head,
    n_layers: int,
    model_device,
    norm_device,
    lm_head_device,
    is_chat: bool = True,
) -> dict:
    """Single forward pass: returns full + exit-layer summed log probs."""
    exit_indices = {
        frac: max(1, min(n_layers - 1, int(n_layers * frac)))
        for frac in exit_fractions
    }

    with torch.no_grad():
        if is_chat:
            message = [{"role": "user", "content": prompt}]
            prefix_ids = tokenizer.apply_chat_template(
                message, add_generation_prompt=True,
                return_tensors="pt", tokenize=True, return_dict=False,
            )[0]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prefix_ids.tolist() + completion_ids])
            prefix_len = prefix_ids.shape[0]
        else:
            prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prompt_ids.tolist() + completion_ids])
            prefix_len = prompt_ids.shape[0]

        if not completion_ids:
            return None

        outputs = model(
            input_ids.to(model_device),
            use_cache=False,
            output_hidden_states=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("output_hidden_states=True produced no hidden states")

        log_probs = outputs.logits.log_softmax(-1).squeeze(0)  # [seq_len, vocab]

        exit_log_probs_map = {}
        for frac, exit_idx in exit_indices.items():
            exit_hidden = outputs.hidden_states[exit_idx]
            exit_hidden_normed = norm(exit_hidden.to(norm_device))
            exit_logits = lm_head(exit_hidden_normed.to(lm_head_device))
            exit_log_probs_map[frac] = exit_logits.log_softmax(-1).squeeze(0)

        full_sum = 0.0
        exit_sums = {frac: 0.0 for frac in exit_fractions}
        n_toks = 0
        for k, tok_id in enumerate(completion_ids):
            t = prefix_len + k - 1
            if t < 0 or t >= log_probs.shape[0]:
                continue
            full_sum += log_probs[t, tok_id].item()
            for frac in exit_fractions:
                exit_sums[frac] += exit_log_probs_map[frac][t, tok_id].item()
            n_toks += 1

        return {
            "num_tokens": n_toks,
            "logp_full": full_sum,
            **{f"logp_exit{int(round(frac*100)):03d}": exit_sums[frac] for frac in exit_fractions},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--v21-dir", required=True, help="Path to data/humaneval/v2.1/ containing per-task CSVs")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--exit-layers", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--resume", action="store_true", help="Skip already-scored rows by reading existing output")
    parser.add_argument("--limit", type=int, default=None, help="Score at most N items (for debugging)")
    args = parser.parse_args()

    v21_dir = Path(args.v21_dir)
    output_path = Path(args.output)

    # Collect all per-task test CSVs (exclude train.csv)
    csv_files = sorted(p for p in v21_dir.glob("*.csv") if p.name != "train.csv")
    print(f"Found {len(csv_files)} per-task test CSVs in {v21_dir}")

    # Load all items
    all_items = []
    for csv_path in csv_files:
        rows = load_v21_csv(csv_path)
        for row in rows:
            row["_csv_path"] = str(csv_path)
        all_items.extend(rows)
    print(f"Total items to score: {len(all_items)}")

    # Build resume set from existing output
    done_keys = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["task_id"], r["row_idx"]))
                except Exception:
                    pass
        print(f"Resuming: {len(done_keys)} items already scored")

    # Load model
    print(f"Loading model {args.model} ...")
    cache_dir = os.environ.get("HF_HUB_CACHE", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()
    print("Model loaded.")

    norm, lm_head, n_layers = _get_model_lm_components(model)
    model_device = get_model_input_device(model, "cuda")
    norm_device = next(norm.parameters()).device
    lm_head_device = next(lm_head.parameters()).device
    print(f"n_layers={n_layers}, model_device={model_device}, norm_device={norm_device}")

    # Score
    n_scored = 0
    n_skipped = 0
    n_empty = 0
    with open(output_path, "a") as out_f:
        for item in tqdm(all_items):
            task_id = item.get("task_id", "")
            row_idx = item["_row_idx"]
            key = (task_id, row_idx)

            if key in done_keys:
                n_skipped += 1
                continue

            if args.limit and n_scored >= args.limit:
                break

            prompt, completion = build_prompt_and_completion(item)
            scores = score_item(
                prompt, completion, model, tokenizer,
                args.exit_layers, norm, lm_head, n_layers,
                model_device, norm_device, lm_head_device,
                is_chat=True,
            )
            if scores is None:
                n_empty += 1
                continue

            record = {
                "task_id": task_id,
                "row_idx": row_idx,
                "correct": item.get("correct", ""),
                "model": item.get("model", ""),
                "strategy": item.get("strategy", ""),
                "temperature": item.get("temperature", ""),
                **scores,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_scored += 1

    print(f"\nDone. Scored: {n_scored}, Skipped (resume): {n_skipped}, Empty: {n_empty}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
