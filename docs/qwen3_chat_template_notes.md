# Qwen3 / Qwen3.5 Chat Template Support — What's Wired and What Isn't

Last updated: 2026-05-20

## TL;DR

We added a `disable_thinking` knob to every "active" training and eval code
path so that Qwen3.5 (and any Qwen3 post-trained model) does NOT silently
activate hybrid reasoning mode in `apply_chat_template`. This was the
fix in commit `f9752dd6` (`qwen3.5: thread enable_thinking=False through
chat templates`).

Several less-active or legacy scripts were intentionally NOT retrofitted.
This note documents those gaps so we can find them quickly if they ever
matter.

## Background — why this matters

Qwen3 / Qwen3.5 post-trained tokenizers default to "thinking mode" in
`apply_chat_template`. With no kwargs, the rendered prompt ends with
`<|im_start|>assistant\n<think>\n`, signaling the model to start a
reasoning trace. We do NOT evaluate reasoning models, so this would
mismatch the prompt format we use for everything else (Gemma, Llama,
Qwen2.5).

The fix is to pass `enable_thinking=False` to `apply_chat_template`,
which renders the prompt with an empty `<think>\n\n</think>\n\n` block,
signaling "thinking already done, give the answer." For non-Qwen3 models,
the kwarg is simply not passed (the call is byte-identical to before),
verified by `scripts/_check_disable_thinking_smoke.py`.

Detection mirrors the existing `_qwen3_post_trained` heuristic:
```python
_qwen3_post_trained = (
    re.search(r'[Qq]wen3', model_name) is not None
    and 'Base' not in model_name
)
```

## What IS wired (covered by `disable_thinking`)

These scripts detect Qwen3+ post-trained models AND thread
`disable_thinking=...` through every `apply_chat_template` call site
and every `utils.get_*` helper they use:

- `src/utils.py` — `get_final_logit_prob`, `get_response`,
  `get_completion_token_logprobs`, `get_completion_token_logprobs_exit`
  all accept `disable_thinking=False` (default no-op).
- `scripts/ranking_loss_ref.py` (main training)
- `scripts/ranking_loss_ref_gemma4.py` (gemma-4 training variant)
- `scripts/ranking_loss_ref_online.py`
- `scripts/ranking_loss_ref_experimental.py`
- `scripts/eval_by_claude.py`
- `scripts/eval_exit_layers.py`

## What IS NOT wired — and why

### 1. Real gap (would silently break Qwen3.5 if exercised)

**`scripts/run_v21_exit_eval.py`** (line ~83)
- Hardcodes `is_chat=True` and accepts `--model` (default
  `google/gemma-4-31B-it`). If pointed at a Qwen3.5 model, the chat
  template will activate thinking mode.
- No `_qwen3_post_trained` detection in the script.
- **If you ever run this with Qwen3.5, retrofit it first.** Mirror the
  pattern in `eval_exit_layers.py`: detect `_qwen3_post_trained`, set
  `model_disable_thinking`, thread `disable_thinking=...` into
  `apply_chat_template` (line 83).

### 2. Broader gap — Qwen3.5 not even detected as a chat model

These scripts use the older `'instruct' in name or '-it' in name`
heuristic, which returns False for `Qwen/Qwen3.5-4B` / `Qwen/Qwen3.5-9B`
(the names contain neither). So they would run Qwen3.5 as if it were a
base model and never call `apply_chat_template` at all — meaning
`disable_thinking` wouldn't help, and the broader fix is to update the
chat-detection heuristic to also recognize Qwen3+ post-trained models
(see how `eval_by_claude.py:855-865` does it).

If you ever want to use these scripts with Qwen3.5, you must (a) update
the chat-detection block to include `_qwen3_post_trained`, AND (b) add
`disable_thinking` plumbing in the same pattern as the active scripts.

- `scripts/eval.py` (line ~436)
- `scripts/validate_validator_log_probs.py` (line ~175)
- `scripts/eval_disc_agreememt.py` (line ~117)
- `scripts/consistency_ft.py` (lines ~208, ~282)
- `scripts/ranking_loss_ref2.py` (legacy training variant)
- `scripts/ranking_loss_ref_explore.py` (legacy training variant)

### 3. Truly safe to ignore

These scripts cannot be exercised with Qwen3.5 in their current state:

- `scripts/analyze_yes_no_distribution.py:90` — hardcodes `is_chat=False`.
- `scripts/logodds.py:156` — hardcodes `is_chat=False` (with a TODO).
- `scripts/_show_persona_prompts.py` — hardcodes
  `google/gemma-2-2b-it` tokenizer (visualization tool only).
- `wordfreq_gpt2_analysis.py:225` — hardcodes
  `model_name = "google/gemma-2-2b-it"`.
- `scripts/dataset_builder/build_gsm8k_dataset.py` — has a local
  `get_response` function that is unrelated to `utils.get_response`.
- `scripts/fine_tune_lora.py:274`,
  `scripts/consistency_ft.py:384` — `apply_chat_template` lines are
  commented out.
- Probe scripts under `scripts/_check_qwen35_*.py` and
  `scripts/_check_disable_thinking_smoke.py` — by design they test the
  raw template behavior, so they should NOT route through
  `disable_thinking`.

## How to retrofit one of the gap scripts (if/when needed)

Pattern (mirrors what's already in `eval_by_claude.py`):

```python
import re

_name_looks_instruct = (
    'instruct' in modelname.lower() or '-it' in modelname.lower()
)
_qwen3_post_trained = (
    re.search(r'[Qq]wen3', modelname) is not None
    and 'Base' not in modelname
)
if _name_looks_instruct or _qwen3_post_trained:
    model_is_chat = True

if 'llama' in modelname.lower() or 'qwen' in modelname.lower():
    model_has_system_role = True

# NEW: thread this through apply_chat_template + utils helpers
model_disable_thinking = _qwen3_post_trained
chat_template_kwargs = (
    {"enable_thinking": False} if model_disable_thinking else {}
)
```

Then thread `**chat_template_kwargs` into every local
`apply_chat_template` call, and pass
`disable_thinking=model_disable_thinking` to every
`get_completion_token_logprobs` / `get_final_logit_prob` /
`get_response` / `get_completion_token_logprobs_exit` /
`get_completion_logprobs_all_exits` call in that file.

For backwards-compat verification, run:

```bash
python scripts/_check_disable_thinking_smoke.py
```

It checks that for non-Qwen3 tokenizers, the new path is byte-identical
to the old call — if that fails, you've broken something.
