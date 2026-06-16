# Generator-context sizing fix — qwen-3.5 HumanEval training only

**Date:** 2026-06-15
**Scope:** Affects ONLY the qwen-3.5-9B HumanEval training run. Nothing else.

## TL;DR
Training Qwen/Qwen3.5-9B on HumanEval aborted with a truncation error. The fix lives
in an **isolated copy** of the trainer — `scripts/ranking_loss_ref_fix_genctx.py` — used
**only** by the qwen HumanEval launcher. The shared trainer
`scripts/ranking_loss_ref_fix.py` is **byte-for-byte unchanged**, so the gemma-4 run and
every other experiment are completely unaffected.

## The problem
`ranking_loss_ref_fix.py` picks one fixed sequence length, `max_context_length`, to
encode/pad every training sequence, and computes it from the **discriminator** prompts
only (~line 778). The **generator** sequence it then encodes (`prompt + completion`,
line ~1690) has a different prompt shape and can be longer. When it exceeds
`max_context_length`, the completion tail is truncated and the trainer's integrity guard
`_check_tail` (line ~1717) **correctly aborts** rather than train on corrupted data:

```
ValueError: generator j tail mismatch (likely truncation/tokenization mismatch).
```

Measured: qwen `max_context_length` = 674, gemma = 682. On the same HumanEval data,
gemma had ~8 tokens of headroom over its longest generator sequence; qwen's tokenizer
produces slightly more tokens and landed just under, so one generator sequence ran past
674. Zero-shot was already in use on both sides (verified) — this is purely a context
*sizing* mismatch, not a shots/prompt-length issue.

### Why it never surfaced before
First time this trainer trained on HumanEval (the long-completion task) with the qwen
tokenizer and no `--max-seq-len`. Other recent tasks had short completions
(membership/rosch/persona) or used `--max-seq-len` (ifeval, which drops over-long items).
The latent bug was always there; every prior run stayed on the safe side of it. The
`_check_tail` guard that turns it into a loud crash (vs silent truncation) is part of the
`fix1` trainer, so older pre-`fix1` runs wouldn't even have errored.

## The fix
One **additive** block in the copy, right after `max_context_length` is computed: also
measure the generator sequence lengths (reusing the trainer's own `_encode_len` logic
from the `--max-seq-len` filter) and raise the context to `max(disc, gen)`.

```python
# size max_context_length to also cover the generator side
_gen_max = max(_gen_seq_len(it) for it in L_train)
if _gen_max > max_context_length:
    max_context_length = _gen_max   # grows only; drops nothing; no-op when gen <= disc
```

Properties:
- **Drops no data** (contrast with `--max-seq-len`, which would drop the long items).
- **Never truncates** — the context now fits every sequence that gets encoded.
- **No-op for gemma** and any case where the generator is already ≤ the discriminator
  max, so it is safe to reuse beyond qwen if desired later.

The diff between the copy and the shared trainer is exactly this block and nothing else
(verified with `diff`).

## What uses the copy (and what does NOT)
| Run | Trainer | Affected by this change? |
|---|---|---|
| qwen-3.5-9B HumanEval (s2/s4 × upper/multi) | `ranking_loss_ref_fix_genctx.py` | YES — this is the only consumer |
| gemma-4-31B-it HumanEval (live run) | `ranking_loss_ref_fix.py` | NO — shared trainer unchanged |
| all qwen/gemma-2 reruns (ifeval, membership, rosch, …) | `ranking_loss_ref_fix.py` | NO |
| the paper's existing results | `ranking_loss_ref_fix.py` | NO |

Launcher pointing at the copy: `scripts/mll_qwen35_he_train_eval.sbatch`
(`python ranking_loss_ref_fix_genctx.py ...`). Commit: `57432158`.

## Status
Smoke job 44979 launched 2026-06-15 ~23:12 CDT to validate the fix end-to-end (train →
merged dir → eval → clean scores) before launching the four full jobs
(cu/cm × s2/s4). The smoke log should print
`[gen-context fix] Raising max_context_length 674 -> <N>` confirming the bump.

## If we later want to upstream
This block is a genuine bugfix (the shared trainer should size context to fit the
generator, not just the discriminator). It is isolated in the copy for now purely to
keep the live gemma run + paper results byte-identical. Upstreaming into
`ranking_loss_ref_fix.py` is the maintainer's call — it is a no-op for every run that
already passed, so it would not change existing results, only prevent the abort.
