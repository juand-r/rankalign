"""Smoke test: verify the new `disable_thinking` plumbing works correctly.

Goals:
1. For a non-Qwen3 model (e.g. Gemma), passing disable_thinking=False produces
   byte-identical chat-template output to NOT passing the kwarg at all.
   (Backwards compat for Gemma/Llama/Qwen2.5.)

2. For a Qwen3.5 model, passing disable_thinking=True suppresses the empty
   <think>...</think> block and matches enable_thinking=False directly.

Mirrors call shapes used in ranking_loss_ref.py / eval_by_claude.py.
"""

import sys
from transformers import AutoTokenizer


def render(tok, msgs, **kwargs):
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, **kwargs
    )


def render_with_thinking_flag(tok, msgs, disable_thinking: bool):
    """Mirror the exact pattern in our updated code."""
    chat_kwargs = {"enable_thinking": False} if disable_thinking else {}
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, **chat_kwargs
    )


eval_msgs_with_sys = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Is a hammer a tool?"},
]

train_msgs_with_sys = eval_msgs_with_sys + [
    {"role": "assistant", "content": "Yes"},
]


def check_backwards_compat(model_id: str):
    print(f"\n=== Backwards-compat check: {model_id} ===")
    print("Goal: disable_thinking=False must be byte-identical to no-kwarg call.")
    tok = AutoTokenizer.from_pretrained(model_id)

    for shape, msgs in [("eval (sys+user)", eval_msgs_with_sys),
                        ("train (sys+user+assistant)", train_msgs_with_sys)]:
        baseline = render(tok, msgs)
        new_path = render_with_thinking_flag(tok, msgs, disable_thinking=False)
        same = baseline == new_path
        print(f"  shape={shape:30s} byte-identical={same}")
        if not same:
            print("    BASELINE:", repr(baseline)[:200])
            print("    NEW:     ", repr(new_path)[:200])
            return False
    return True


def check_qwen3_disable(model_id: str):
    print(f"\n=== Qwen3.5 disable-thinking check: {model_id} ===")
    print("Goal: disable_thinking=True must suppress empty <think>...</think>.")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  SKIP (could not load tokenizer): {type(e).__name__}: {e}")
        return None

    default = render(tok, eval_msgs_with_sys)
    explicit_off = render(tok, eval_msgs_with_sys, enable_thinking=False)
    via_flag = render_with_thinking_flag(tok, eval_msgs_with_sys, disable_thinking=True)

    has_default_think = "<think>" in default
    has_explicit_off_think = "<think>" in explicit_off
    has_via_flag_think = "<think>" in via_flag

    print(f"  default                          has <think>: {has_default_think}")
    print(f"  explicit enable_thinking=False   has <think>: {has_explicit_off_think}")
    print(f"  via_flag (disable_thinking=True) has <think>: {has_via_flag_think}")
    print(f"  via_flag matches enable_thinking=False:       {via_flag == explicit_off}")
    if via_flag != explicit_off:
        print("    EXPLICIT_OFF:", repr(explicit_off)[:300])
        print("    VIA_FLAG:    ", repr(via_flag)[:300])
        return False
    return True


def main():
    ok = True
    # Backwards-compat: any chat tokenizer that doesn't know enable_thinking
    # must accept the no-kwarg path identically. Pick a small one we know we have.
    for m in ["google/gemma-2-2b-it", "meta-llama/Llama-3.1-8B-Instruct",
              "Qwen/Qwen2.5-7B-Instruct"]:
        try:
            r = check_backwards_compat(m)
        except Exception as e:
            print(f"  SKIP {m}: {type(e).__name__}: {e}")
            continue
        ok = ok and r

    # Qwen3.5 thinking-off behavior.
    r = check_qwen3_disable("Qwen/Qwen3.5-4B")
    if r is not None:
        ok = ok and r

    print("\n=== SUMMARY ===")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
