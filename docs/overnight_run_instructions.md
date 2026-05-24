# Overnight Run Instructions (2026-05-24)

User instructions, recorded verbatim from the chat at 2026-05-24 03:14am
(US/Central) so I (the agent) can refer back to them while the user is
asleep. The user said "do exactly this; failure is not an option;
don't wait for me to wake up."

## Rules of engagement

1. **Launch / cancel freely.** A job stalled? Cancel. A job failed?
   Relaunch. Realised something is broken? Fix it. The user trusts me.
2. **Failure is not an option.** Treat this like an exam: get as
   many train+eval pairs as possible. If stuck on one, move on.
3. **Don't wait.** The user is asleep. Be self-sufficient.
4. **Save models on /datastor2** (the launch rule re: relative paths).

## Job list (priority order)

Train each setting on each (model, dataset) cell.

**Most important**: s4, s7, s2, s3, s1
**Less important (only if time)**: s5, s6, s11, s12

Settings 1-12 are defined in
[`docs/IMPORTANT-RESEARCH-PLAN.md`](IMPORTANT-RESEARCH-PLAN.md) §2:

| # | Setting          | Loss      | Semi/lo   | log-odds | force-same-x | TC   |
|---|------------------|-----------|-----------|----------|--------------|------|
| 1 | SFT-lo           | sft       | labelonly | —        | —            | —    |
| 2 | RankAlign        | pref-only | semi      | —        | —            | —    |
| 3 | New+fsx          | comb      | semi      | ✓        | ✓            | —    |
| 4 | New+fsx+tc       | comb      | semi      | ✓        | ✓            | self |
| 5 | RankAlign+fsx+tc | pref-only | semi      | — (NEW)  | ✓            | self |
| 6 | RankAlign+tc     | pref-only | semi      | —        | —            | self |
| 7 | New+fsx+negtc    | comb      | semi      | ✓        | ✓            | neg  |
| 11| New+tc           | comb      | semi      | ✓        | —            | self |
| 12| New+negtc        | comb      | semi      | ✓        | —            | neg  |

Note s5: per IRP §1, the historical `--validator-log-odds` flag was a
bug (vlo belongs only with comb). New runs of s5 drop it.

## Models

- `google/gemma-2-2b-it`
- `google/gemma-2-9b-it`

(Note: only the `-it` variants. Not the base `-2b` / `-9b`.)

## Datasets (train → eval)

| # | Train task               | Eval tasks                                                                                         |
|---|--------------------------|----------------------------------------------------------------------------------------------------|
| a | `membership-sans-rosch-v0` | 10 rosch tasks: rosch-{bird, carpenters-tool, clothing, fruit, furniture, sport, toy, vehicle, vegetable, weapon} |
| b | `persona-v1`             | 6 persona-v1 tasks: psychopathy, machiavellianism, narcissism, desire-to-create-allies, interest-in-music, interest-in-science (in-domain + ood) |
| c | `ifeval-concat`          | 21 test-only ifeval tasks: `ifeval-prompt_1` .. `ifeval-prompt_21`                                |

## Training flags (always)

- Use `scripts/ranking_loss_ref_fix.py` (the v7 / fix1 path).
- `--delta-bins 10`
- `--disc-shots few` (NOT zero; the user explicitly said few.)
- For fsx settings (s3, s4, s5, s7): also pass `--per-prompt-delta`
  AND `--shape-budget-mode global`
- `--lora` (auto-added by `run_train_semi.sh` for non-2b models; we
  only have 2b-IT here, which IS a LoRA case — wait: see below).
- `--no-upload-hf` (no need to push tonight).
- `--no-wandb`.
- `--total_samples 5110` (default, fine).
- `--num_epochs 3` (default).

For ifeval-concat with 9b-it use 2 GPUs (model parallelism). Other
combos use 1 GPU.

### LoRA decision (2b-it specifically)

`run_train_semi.sh` line 187 currently does:

```bash
if [[ "$MODEL" != *"-2b"* && "$MODEL" != *"-2b-"* ]]; then
    LORA_FLAG="--lora"
fi
```

So for `google/gemma-2-2b-it` (which contains `-2b-` substring) LoRA
is NOT added — it does full-finetune. For `google/gemma-2-9b-it` LoRA
IS added. Keep this behavior.

## Slurm budget

- Up to 20 RUNNING + 12 PENDING = 32 jobs total at any time.
- No hard wall-time per training job, but smaller asks schedule faster.
- Never run on the login node.

## Failure recovery

- Resubmit on transient slurm failure (node down, OOM, network).
- Re-run evals if scores didn't get written.
- If a job crashes with a real Python error: diagnose, fix it if
  possible, test the fix, re-run. Don't wait for the user.
- Diagnose, course-correct, keep going. Failure is not an option.

## Notes

- I (agent) keep markdown notes in `docs/` as I go.
- Every sbatch / `run` command I submit goes into the notes verbatim
  (for future reproducibility & debugging).

## Out of scope (do NOT touch)

- humaneval anything
- pod-based runs (s1-s12 launchers in
  `pod-setup-train-scripts-gemma-4/`)
- humaneval-correct-multi / humaneval-correct-upper

## Pre-existing reference jobs (running)

- **41730**: membership-sans-rosch-v0 baseline (gemma-2-2b, ppd OFF,
  shape-budget-mode=per-prompt by default). Walltime 5h, hits ~04:35
  US/Central. See `docs/job41730_baseline_reference.md`.
- **41745**: smoke for `--shape-budget-mode global` on the same task.
  PENDING (priority). When it runs it will also exercise the new
  per-item disc-forward gating perf fix.
