"""Show the *exact* model input that eval_by_claude.py will produce
for persona-v0 tasks, for both base and chat-wrapped (gemma-2-*-it) modes.

Prints, for one yes-item and one no-item drawn from the psychopathy test split:
  - generator (zero-shot), raw prompt
  - generator (zero-shot), chat-template wrapped (gemma-2-2b-it tokenizer)
  - discriminator (zero-shot), raw + chat-wrapped
  - discriminator (few-shot),  raw + chat-wrapped

The chat wrapping reproduces what utils.get_final_logit_prob / get_completion_token_logprobs
do for is_chat models without system role (the gemma-2 family).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from transformers import AutoTokenizer

import tasks  # triggers all task registration
from task_registry import TASK_REGISTRY


def show(label: str, text: str) -> None:
    bar = "─" * 78
    print(f"\n{bar}\n{label}\n{bar}\n{text}")
    print(f"  [length: {len(text)} chars]")


def chat_wrap(tokenizer, raw_prompt: str) -> str:
    """Reproduce what is_chat=True path does (gemma-2 family, no system role)."""
    msg = [{"role": "user", "content": raw_prompt}]
    ids = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, return_tensors=None, tokenize=True
    )
    return tokenizer.decode(ids, skip_special_tokens=False)


def main() -> None:
    cfg = TASK_REGISTRY["persona-v0-psychopathy"]
    _, L_test = cfg["load_data"](seed=0)

    # Pick one yes and one no item
    yes_item = next(r for r in L_test if r["correct"] == "yes")
    no_item = next(r for r in L_test if r["correct"] == "no")

    print("=" * 78)
    print("Loading gemma-2-2b-it tokenizer for chat-template demo "
          "(skips weights, just for chat formatting)...")
    print("=" * 78)
    chat_tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    for label, item in [("YES item (correct=yes, persona-matching)", yes_item),
                        ("NO item (correct=no, anti-persona)", no_item)]:
        print("\n" + "█" * 78)
        print(f"█ {label}")
        print(f"█ persona     : {item['persona']}")
        print(f"█ statement   : {item['statement']!r}")
        print(f"█ correct     : {item['correct']}  | label_conf: {item['label_confidence']:.3f}")
        print("█" * 78)

        # ---- generator zero ----
        gen = cfg["make_prompt"](item, style="generator", shots="zero")
        show("GEN (zero-shot) — RAW prompt sent to gemma-2-2b (base)",
             f"PROMPT:     {gen.prompt!r}\n"
             f"COMPLETION: {gen.completion!r}\n"
             f"-> scored: log P(completion | prompt) under the base model")
        show("GEN (zero-shot) — CHAT-WRAPPED prompt sent to gemma-2-*-it",
             chat_wrap(chat_tok, gen.prompt) + gen.completion)

        # ---- disc zero ----
        disc_z = cfg["make_prompt"](item, style="discriminator", shots="zero")
        show("DISC (zero-shot) — RAW prompt for base model",
             f"PROMPT:     {disc_z.prompt!r}\n"
             f"COMPLETION: {disc_z.completion!r}\n"
             f"-> scored: log P(' Yes' | prompt) and log P(' No' | prompt)")
        show("DISC (zero-shot) — CHAT-WRAPPED prompt for *-it",
             chat_wrap(chat_tok, disc_z.prompt))

        # ---- disc few ----
        disc_f = cfg["make_prompt"](item, style="discriminator", shots="few")
        show("DISC (few-shot) — RAW prompt for base model",
             disc_f.prompt + "\n[completion: " + disc_f.completion + "]")
        show("DISC (few-shot) — CHAT-WRAPPED prompt for *-it",
             chat_wrap(chat_tok, disc_f.prompt))


if __name__ == "__main__":
    main()
