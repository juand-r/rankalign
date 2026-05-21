"""One-off: confirm what Qwen3.5 chat template emits.

We need to know:
1. With our current call shape (no enable_thinking kwarg), does the template
   inject a <think> opening tag?
2. With enable_thinking=False, does it close <think></think> immediately?
3. Does enable_thinking=False even work (does it raise / get ignored)?

Mirrors the call pattern in ranking_loss_ref.py:1519:
    apply_chat_template(messages, add_generation_prompt=True, ...)
and the eval-time call pattern in eval_by_claude.py:494.
"""

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"

tok = AutoTokenizer.from_pretrained(MODEL)

# Mirror ranking_loss_ref.py shape: system + user + assistant (training)
ms_tune = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Is a hammer a tool?"},
    {"role": "assistant", "content": "Yes"},
]

# Mirror eval shape: system + user (then add_generation_prompt=True)
ms_eval = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Is a hammer a tool?"},
]


def show(label, **kwargs):
    print(f"\n--- {label} ---")
    print(f"kwargs: {kwargs}")
    try:
        out = tok.apply_chat_template(ms_eval, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return
    print("Rendered text:")
    print(repr(out))
    print("Has <think>:", "<think>" in out)
    print("Has </think>:", "</think>" in out)


print(f"Model: {MODEL}")
print(f"Tokenizer chat template length: {len(tok.chat_template) if tok.chat_template else 0} chars")

# Default (no enable_thinking arg) — this is what our code currently does.
show("default (no enable_thinking)")

# Explicit True
show("enable_thinking=True", enable_thinking=True)

# Explicit False
show("enable_thinking=False", enable_thinking=False)

# Also check the assistant-message training shape, where add_generation_prompt=True
# is set after a finished assistant turn.
print("\n\n=== TRAINING SHAPE (system+user+assistant, add_generation_prompt=True) ===")
print("This is what ranking_loss_ref.py does at line 1519.")
for label, kw in [
    ("default", {}),
    ("enable_thinking=False", {"enable_thinking": False}),
]:
    print(f"\n--- training, {label} ---")
    out = tok.apply_chat_template(ms_tune, tokenize=False, add_generation_prompt=True, **kw)
    print(repr(out))
    print("Has <think>:", "<think>" in out)
    print("Has </think>:", "</think>" in out)
