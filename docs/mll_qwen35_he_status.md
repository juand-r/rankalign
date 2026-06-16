# Qwen3.5-9B humaneval (s2/s4 × upper/multi) — status / BLOCKED pending decision

**Goal:** Qwen/Qwen3.5-9B LoRA train+eval on humaneval-v2.1correct-{upper,multi},
settings s2 (RankAlign) + s4 (New+fsx+self-tc) — the qwen sibling of the gemma-4 run.
Outputs: models2-rerun-wandb / outputs-rerun-wandb. Metrics wanted (task #26):
Pearson corr, gen ROC, val acc, val ROC.

## Launcher
`scripts/mll_qwen35_he_train_eval.sbatch` (committed) — port of the gemma-4 humaneval
sbatch with qwen specifics (Qwen3.5-9B, qwen35 venv + moe.py patch, plain --lora ->
merged dir, 2× A40, rerun-wandb dirs). s2 = option (a): NO --validator-log-odds.

## Smoke history
- Smoke 44977 FAILED (7s): `set -euo pipefail` aborted on an unguarded `ls|head` in
  `adapter_dir_for`. FIXED (end in `echo`, guard with `|| true`). Commit 84690600.
- Smoke 44978 FAILED (9.5 min, mid-training step 10/65): see blocker below.

## BLOCKER (needs user decision)
`ValueError: generator j tail mismatch (likely truncation/tokenization mismatch)` —
ranking_loss_ref_fix.py `_check_tail` (line 1717).

Root cause (verified): `max_context_length` is computed from the **discriminator**
chat-templated prompts only (ranking_loss_ref_fix.py ~778). qwen=674, gemma=682 tokens.
The **generator** sequence (`prompt_j + completion_j`, line 1690) is then encoded with
`max_length=max_context_length`. For qwen, ≥1 humaneval generator sequence exceeds 674,
so its completion tail is truncated and the trainer's own `_check_tail` aborts (correct
behavior — refuses to train on truncated data). Gemma had an 8-token margin so it never
tripped. qwen `model_max_length`=262144, so not a model limit.

### Options (NOT yet actioned)
1. **(Recommended) Additive trainer fix:** size `max_context_length` to also cover the
   generator sequences (`max(disc, gen)`). Purely *grows* the encode length to fit what
   is already encoded -> no data dropped, no truncation; no-op for gemma (gen ≤ disc there).
   Apply as an ISOLATED patched copy of the trainer for qwen-he so the shared trainer +
   the live gemma run stay byte-identical. Smoke-validate, then launch.
2. `--max-seq-len N`: only caps DOWNWARD + drops items exceeding N (silent data loss).
   Does NOT cleanly fix this (can't raise the 674 encode length). NOT recommended.
3. Investigate the specific over-long item(s) first.

## State as of 2026-06-15 ~22:55 CDT
- Full run NOT launched (would fail identically). `.q35_full_launched` marker absent.
- qwen auto-driver cron (4308d4e7) DELETED to stop the resubmit loop.
- gemma-4 run + its hourly monitor (cron 25cebb68) untouched and progressing.
- Resume: pick an option above; if (1), patch the copy, `SMOKE=1` smoke s2, validate
  scores, then launch 4 jobs (cu/cm × s2/s4) and re-arm a monitor cron.
