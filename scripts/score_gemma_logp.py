"""Score completions with gemma-2-9b-it: log P(completion | prompt).

Reads a JSONL of (task_id, model, strategy, temperature, correct, prompt, completion)
rows and adds per-row:
  - log_prob: float, sum of log-probabilities of completion tokens given the prompt
  - num_completion_tokens: int, number of tokens in the completion (assigned positions)
  - log_prob_per_token: float = log_prob / num_completion_tokens

The completion is tokenized in context of the prompt (i.e., the prompt and the
completion are concatenated, tokenized once, and we measure log-prob only on
the completion's token positions). This avoids tokenization boundary artifacts.

Resumable: rows already present in the output (matched by task_id + model +
strategy + temperature + prompt hash) are skipped.

Why this script existed before but wasn't committed: the v1 gemma-2-9b-it
scoring run produced notes/gsm8k-v1-gemma2-scoring/data/v1_scores_gemma-2-9b-it.jsonl
via ad-hoc code in /tmp. Reconstructed here so v1.1 scoring is reproducible.

Usage:
  python score_gemma_logp.py \\
      --input  notes/.../v1_score_inputs.jsonl \\
      --output notes/.../v1.1_scores_gemma-2-9b-it.jsonl \\
      [--model google/gemma-2-9b-it] [--batch-size 1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "google/gemma-2-9b-it"


def _row_key(r: dict) -> str:
    """Stable key for resumability."""
    h = hashlib.md5()
    for k in ("task_id", "model", "strategy", "temperature"):
        h.update(str(r.get(k, "")).encode("utf-8"))
        h.update(b"\0")
    h.update((r.get("prompt") or "").encode("utf-8"))
    return h.hexdigest()


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(_row_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return seen


@torch.inference_mode()
def score_row(prompt: str, completion: str, model, tokenizer, device) -> tuple[float, int]:
    """Return (sum_log_prob, num_completion_tokens) of completion given prompt.

    Implementation: tokenize prompt and prompt+completion separately, then
    compute logprobs only over the positions corresponding to completion
    tokens (positions [prompt_len .. full_len)). Uses standard causal-LM
    shift-by-one (predict token t from logits at position t-1).
    """
    # Tokenize prompt alone (no special tokens added on top — we want the
    # prompt as the leading prefix of the full input below).
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]
    if full_len <= prompt_len:
        return 0.0, 0
    full_ids = full_ids.to(device)
    logits = model(full_ids).logits  # (1, full_len, vocab)
    # Shift: position t's logits predict token t+1
    shift_logits = logits[0, :-1, :]
    shift_labels = full_ids[0, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)  # (full_len-1, vocab)
    # Completion token positions in shifted space: [prompt_len-1 .. full_len-1)
    # i.e., the token at full position prompt_len is predicted by logits at prompt_len-1.
    comp_log_probs = log_probs[prompt_len - 1 : full_len - 1].gather(
        1, shift_labels[prompt_len - 1 : full_len - 1].unsqueeze(-1)
    ).squeeze(-1)
    return float(comp_log_probs.sum().item()), int(comp_log_probs.numel())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input JSONL: task_id, model, strategy, temperature, correct, prompt, completion")
    ap.add_argument("--output", required=True, help="Output JSONL (resumable)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Scoring model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows scored (debug).")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = _existing_keys(out)
    print(f"[score] {len(seen):,} rows already scored in {out}", flush=True)

    print(f"[score] loading {args.model} ...", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    print(f"[score] ready on {device}", flush=True)

    n_in = 0
    n_done = 0
    t0 = time.time()
    with inp.open() as fin, out.open("a") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            k = _row_key(r)
            if k in seen:
                continue
            prompt = r.get("prompt", "")
            completion = r.get("completion", "")
            log_prob, n_tok = score_row(prompt, completion, model, tokenizer, device)
            r["log_prob"] = log_prob
            r["num_completion_tokens"] = n_tok
            r["log_prob_per_token"] = (log_prob / n_tok) if n_tok else 0.0
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            seen.add(k)
            n_done += 1
            if n_done % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"[score] scored {n_done} (read {n_in}) — {elapsed:.0f}s — {n_done / elapsed:.2f} rows/s", flush=True)
            if args.max_rows is not None and n_done >= args.max_rows:
                break

    elapsed = time.time() - t0
    print(f"[score] DONE — scored {n_done} rows in {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
