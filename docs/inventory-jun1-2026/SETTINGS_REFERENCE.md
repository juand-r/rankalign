# Settings Reference — RankAlign training variants (s1–s13)

**Compiled:** 2026-06-01 (inventory take-stock pass)
**Purpose:** Canonical definition of what each numbered/`s`-prefixed setting means, so
TASK1 (training inventory) and TASK2 (eval inventory) have one source of truth.

**Verified from source** (not reconstructed from memory):
- `pod-setup-train-scripts-gemma-4/run_settings_v21correct_upper.sh` — gemma-4 numbered settings (humaneval correct-upper/multi)
- `pod-setup-train-scripts-gemma-4/TRAINING_PLAN_v21correct_multi.md` — the 10-setting diagnostic plan
- `scripts/_overnight_launch.sh` — gemma-2 / qwen3.5 `s`-settings dispatcher (membership/persona/ifeval/humaneval)
- `docs/s13_consistency_ft.md` — s13 definition

---

## Loss families (the key axis)

Every setting is one of three loss families, set by the `(pref, nllv, nllg)` weight triple:

| Family | pref_w | nllv_w | nllg_w | Meaning |
|--------|--------|--------|--------|---------|
| **SFT** | 0 | 1 | 1 | Pure supervised NLL (no preference/ranking term). Settings **s1**, **s13**. |
| **RankAlign** (a.k.a. pref-only) | 1 | 0 | 0 | Preference/ranking loss only. Settings **s2, s5, s6, s8, s9, s10**. |
| **comb** (a.k.a. "New") | 1 | 1 | 1 | Combined preference + NLL. Settings **s3, s4, s7, s11, s12**. |

Other axes layered on top:
- **fsx** = `--force-same-x` — within-prompt pair construction. **Standalone flag with no
  dependency** — fsx *alone* (without ppd/sbm-global) is valid in the training code.
- **ppd** = `--per-prompt-delta` and **sbm-global** = `--shape-budget-mode global` — these
  **require `--force-same-x`** (and ppd additionally requires `--delta-bins N`). Enforced by
  `parser.error` in `scripts/ranking_loss_ref_fix.py` L2510–2521. So the implication runs
  **ppd / sbm-global ⟹ fsx**, *not* the reverse.
  - **Caveat:** our launchers (`run_settings_v21correct_upper.sh`, `_overnight_launch.sh`)
    always add `--per-prompt-delta --shape-budget-mode global` whenever a setting uses fsx, so
    in *practice* every fsx run here is also ppd+sbm-global. That is a launcher convention, not
    a code constraint. Empirically: **all 88 v7 fsx model dirs also have `ppd`; 0 are fsx-only**
    (and 0 are ppd-without-fsx). If you want an fsx-only ablation, the code permits it — no such
    run exists yet.
- **vlo** = `--validator-log-odds` (train-time). Present in comb settings; **deliberately OFF** in pref-only+fsx+tc settings (s5/s8) — that was a historical gemma-2 bug, fixed for v7.
- **train-time TC** = `--self-typicality` / `--neg-typicality` during training (s4, s5, s6, s7, s8, s9, s11, s12).
- **cft** = `--consistency-ft` — drops labeled items where binarized validator/generator scores disagree (s13 only).
- **semi** = `--semi-supervised 0.1` (default) vs `--labeled-only 0.1` (s1, s13).

---

## The numbered settings (gemma-4 humaneval: correct-upper & correct-multi)

Source: `run_settings_v21correct_upper.sh::configure_setting()`. COMMON flags:
`--num_epochs 3 --train_g_or_d g --split_type random --all --delta 0.15 --disc-shots zero --lora --gradient_checkpointing --total_samples 5110`, plus **`--delta-bins 10 --gemma4-lora`** at call site (so realized delta is auto-computed, NOT 0.15 — see Delta note).

| # | Name | Loss | fsx | vlo | train-TC | semi | **Eval modes** |
|---|------|------|-----|-----|----------|------|----------------|
| 1 | SFT-lo | SFT | – | – | – | labeled-only 0.1 | self + neg (both base-TC) |
| 2 | RankAlign | pref-only | – | – | – | semi 0.1 | self + neg |
| 3 | New+fsx | comb | ✓ | ✓ | – | semi 0.1 | self + neg |
| 4 | New+fsx+tc | comb | ✓ | ✓ | self | semi 0.1 | self |
| 5 | RankAlign+fsx+tc | pref-only | ✓ | ✗(fixed) | self | semi 0.1 | self |
| 6 | RankAlign+tc | pref-only | – | – | self | semi 0.1 | self |
| 7 | New+fsx+negtc | comb | ✓ | ✓ | neg | semi 0.1 | neg |
| 8 | RankAlign+fsx+negtc | pref-only | ✓ | ✗(fixed) | neg | semi 0.1 | neg |
| 9 | RankAlign+negtc | pref-only | – | – | neg | semi 0.1 | neg |
| 10 | RankAlign+fsx | pref-only | ✓ | – | – | semi 0.1 | self + neg |
| 11 | New+tc | comb | – | ✓ | self | semi 0.1 | self |
| 12 | New+negtc | comb | – | ✓ | neg | semi 0.1 | neg |
| 13 | SFT + consistency-ft | SFT + cft | – | – | – | labeled-only 0.1 | self + neg |

> **Eval modes** = which typicality correction is applied at eval. All eval runs also
> pass `--base-typicality --base-model <base>` and `--validator-log-odds`, and every CSV
> additionally carries the **raw** (no-TC) score. So the result-table columns are:
> **Raw**, **PMI self** (self-TC), **PMI base** (self-TC+base), **Neg self**, **Neg base**.

### Diagnostic structure (from TRAINING_PLAN)
- Does TC alone help? → #6/#9 vs #2 vs #1
- Does neg-TC alone help? → #9 vs #6 vs #2
- Does fsx alone help? → #10 vs #2
- Does comb+fsx help? → #3 vs #2
- Does TC help on top of comb+fsx? → #4 vs #3 (self), #7 vs #3 (neg)
- Does fsx add anything given TC? → #6 vs #5 (self), #9 vs #8 (neg)

---

## The `s`-settings (gemma-2-2b / 2b-it / 9b-it, qwen3.5-9b: membership/persona/ifeval)

Source: `scripts/_overnight_launch.sh::build_setting()`. Same semantics as the numbered
settings above; this dispatcher covers s1, s2, s3, s4, s5, s6, s7, s11, s12, s13.
COMMON: `--script ranking_loss_ref_fix.py --delta-bins 10`, `--disc-shots few` (gemma-2
default) or `zero` (ifeval/humaneval), fsx settings add `--per-prompt-delta
--shape-budget-mode global`. gemma-4 adds `--gemma4-lora --gradient-checkpointing`.

Datasets handled: `membership-sans-rosch-v0` (eval: 10 rosch tasks), `persona-v1` (eval:
6 persona test tasks), `ifeval-concat` (eval: 21 ifeval-prompt_*), `humaneval-v2.1correct-upper`
(eval: 82 humaneval tasks, gemma-4 only).

**Note:** s8/s9/s10 are NOT in `_overnight_launch.sh` (only in the gemma-4 numbered script).

---

## Delta note (IMPORTANT — affects dir names)

- **v7 (`--delta-bins 10`)**: `--delta 0.15` is just a placeholder. The realized delta is
  auto-computed from the validator-logprob score spread (p5–p95 / 10 bins) at runtime, so
  the **directory name carries the realized delta** (e.g. `delta2.14`, `delta3.25`, `delta0.34`).
  The eval glob matches `delta*` because the value is unknown until training runs.
- **v6**: fixed `delta 0.15` (dir name `delta0.15`).
- **v7b**: fixed `delta 0.15` again (a re-run regime; see memory `v7b_means_delta015`). The
  `outputs_gemma4_from_pod-v7b/` dir names can be misleading — v7b = fixed-delta-0.15, NOT a
  different ID/OOD split.

## Version note

- **v6** = older runs, fixed delta 0.15, `ranking_loss_ref_gemma4.py` (pre-fix), no `--fix1` suffix.
- **v7** = current, `ranking_loss_ref_fix.py`, `--delta-bins 10`, dir/repo names carry `--fix1` suffix.
- **v7b** = v7 code but fixed delta 0.15.

## Naming conventions
- **Local dir** (models2): `v7-google--<model>-delta<D>-epoch<E>--<task>-all--d2g--random--alpha1.0<suffix>` where suffix encodes the flags.
- **Save-dir suffix order** (ranking_loss_ref_fix.py L2182): `{tc}{lenorm}{single}{full-completion}{eos}{pref}{nllv}{nllg}{fsx}{ppd}{cft}{valboost}{vallogodds}{semi}{fix1}`.
- **gemma-4 LoRA** (`--gemma4-lora`): skips `merge_and_unload`, so there is **no `_merged` sibling** — eval targets the adapter dir directly.
- **gemma-2 LoRA (9b-it, 2b-it)**: produces a `_merged` sibling dir (eval target). gemma-2-2b (base, no `-2b-` LoRA rule) trains full.

## Training logs
- Per-run JSON: `<models-dir>/training_run_logs/<timestamp>_<model>_<task>.json` — records delta config, shape-budget, label partition, sampled-pair counts, score stats, seed, and (for s13) the `consistency_ft` filter stats.
- wandb: `WANDB_MODE=offline` on the gemma-4 pods (so wandb logs are local, not on the wandb server) — confirm per run.
