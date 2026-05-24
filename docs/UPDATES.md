# Updates / Status (2026-05-01)

A running, dated log of where things stand on the methods side and what's
worth keeping in mind when planning next steps. Append new dated sections
on top; don't rewrite history.

---

## 2026-05-24 (evening)

### Fix: `OSError: [Errno 36] File name too long` in eval CSV saves

**Symptom.** Job `41982` (s4 / membership-sans-rosch / gemma-2-9b-it) failed
to save score CSVs for all 10 rosch test tasks with `OSError: [Errno 36] File
name too long`. The filename came out as 287+ chars (over Linux's 255-byte
NAME_MAX). Same bug latent for every 9b-it cell with the long
fsx+ppd+vallogodds+tc-self/neg signature (s3, s4, s5, s7).

**Root cause.** `scripts/eval_by_claude.py` had **8 separate** save-to-CSV
sites for the four task-types (ifeval / ambigqa+plausibleqa / membership+rosch
/ registry-based) × the two eval branches (logitlens vs final-layer). Only
**one** of them (the rosch site at the old line ~1610) had an inline filename
truncation. The other 7 used a raw f-string `f".../scores_{prefix}{model_short}_..."`
with `model_short = 'v6-' + modelname.replace('/', '_')`, which embeds the
full `/datastor2/jdr/...` path into the filename and overflows for long
fix1 model dirs.

**Fix.** `scripts/eval_by_claude.py` — switched all 8 save sites to use the
existing shared helper `tasks.common.build_csv_filename()` (which already
calls `build_model_short()` to cap model_short at 160 chars with an 8-char
md5 hash suffix when needed). One commit, ~70 lines deleted, ~30 lines added.
The helper handles absolute paths via `os.path.basename()` instead of
`replace('/', '_')`, which kills the `_datastor2_jdr_...` prefix and saves
~30 chars per filename even before truncation.

**Verified.** For the failing 41982 modelname, `build_csv_filename(...)` now
emits a 216-char filename (well under 255), with the truncated model_short
ending in `..._92c5a9fe` (8-char hash for uniqueness across cells).

**Side effect on table builders.** Because the new pathway no longer prepends
`'v6-'` to absolute-path-derived model_shorts, eval CSVs from re-runs after
this fix will start with `v7-google--gemma-2-...` rather than
`v6-_datastor2_jdr_rankalign_models2_v7-google--gemma-2-...`. The v7 table
builders (`scripts/_build_rosch_table_v7.py`, `_build_persona_v1_table_v7.py`)
use regex patterns that already expect the un-prefixed `v7-...` form, so this
is a *step toward* what they want. (They likely had a separate bug expecting
single-`_` between `google` and `gemma` where the actual format is `--`; that
is a separate issue, not introduced by this fix — flagged here so it isn't
forgotten.)

**Re-runs needed.** Any cell whose Slurm eval job exited with `OSError 36`
needs an eval re-run (model checkpoints are saved on disk; only the eval CSV
write failed). Confirmed cases include 41982. Other cells using
`fsx+ppd+vallogodds+tc-self/neg` on 9b-it likely affected as well.

---

## 2026-05-24 (afternoon)

### New training flag: `--consistency-ft` (and new setting s13)

Added an opt-in SFT-only data filter to `scripts/ranking_loss_ref_fix.py`.
The full design + the 10 launch commands live in
[`docs/s13_consistency_ft.md`](s13_consistency_ft.md). High-level:

**What it does.** Before training:

1. Computes per-item validator scores (already done by the existing pre-pass:
   `logprobs_last_layer[i]`).
2. New forward-only pass to compute per-item generator scores
   `gen_scores[i] = log P(completion_i | gen_prompt_i)` (raw, no length-norm,
   no typicality).
3. Means: `t_v = mean(val)`, `t_g = mean(gen)`. Under `--semi-supervised`,
   means are computed over labeled items only.
4. Binarize each item by its threshold. **Drop** any *labeled* item where
   `bv != bg`. Unlabeled items pass through unchanged.
5. Pair construction + training proceed on the filtered set.

**Argparse enforces** (verified): `--preference_loss_weight 0`,
`--nll_validator_weight > 0`, `--nll_generator_weight > 0`, and
`--force-same-x` OFF. `--labeled-only`, `--semi-supervised`, or neither all
allowed. Off by default → bit-for-bit identical to today for s1–s12.

**Save-dir gets a `--cft` slot** between `--ppd` and `--vallogodds` so
trained adapters from cft runs are unambiguous.
Per-run JSON log gets a `"consistency_ft"` object with
`t_v`, `t_g`, `n_labeled_kept/dropped`, `n_unlabeled_kept`, and the 4-cell
`(bv, bg)` count breakdown.

### New dispatcher setting `s13` and dataset `humaneval`

`scripts/_overnight_launch.sh` was extended:

- **SETTING `s13`** = SFT-lo flag set (s1) + `--consistency-ft`.
  Eval TC list = `self neg`.
- **DATASET `humaneval`** (gemma-4-31B-it only) — task list of 82 dynamically
  enumerated `humaneval-v2.1correct-upper-humaneval_*` from the data dir.
- **Auto-detect of `gemma-4-31B-it`** in the model string switches:
  - `VENV_OVERRIDE=/datastor2/jdr/venvs/gemma4` (transformers 5.x;
    `venv_lexcons` is 4.46.x and won't load Gemma-4).
  - `--gemma4-lora` (regex target_modules; skip `merge_and_unload` →
    eval points at the adapter dir, not `_merged`).
  - `--gradient-checkpointing` (VRAM).
  - `--disc-shots zero` (matches `run_settings_v21correct_upper.sh`).
  - 3 GPUs, 192 GB RAM, 24 h walltime.
- `MERGED_SUFFIX` logic correctly returns "" for `--gemma4-lora` runs.

### Plumbing

- `scripts/run_train_semi.sh`: pass-through for `--consistency-ft` and
  `--gradient-checkpointing`; new `VENV` env var override (default
  `/u/jdr/venvs/venv_lexcons`).
- `scripts/run_eval_semi.sh`: `VENV` env var override.
- The dispatcher injects `export VENV='...'` into the eval `--wrap` so the
  dependency-chained eval picks the right venv.

### Verification (no jobs submitted)

Dryrun (`DRYRUN=1`) of every s13 cell — `persona/membership/ifeval ×
{2b, 2b-it, 9b-it} × s13` plus `humaneval × gemma-4-31B-it × s13` — produces
the expected `Expected save:` glob with `--cft` in the right slot,
`gemma-2-9b-it ifeval` correctly auto-allocates 2 GPUs, gemma-4-31B-it
correctly gets 3 GPUs / venv / disc-shots zero / no `_merged` suffix.
All three argparse guards fire on misuse. `s1/s4/s7` regression-checked:
no spurious `--cft` injected.

### Outstanding TODO

[`docs/TODO_s13_launch.md`](TODO_s13_launch.md) — **launch the 10 train +
20 eval jobs for s13** when the cluster has headroom. Currently held back
because the overnight queue is still working through s1–s12.

---

## 2026-05-02

### Added GSM8K task family (modern pattern)

Two paired families of math word-problem tasks built from RLHFlow's Mistral
generations on GSM8K test:

- `gsm8k-full-*`       — solution ends with "The answer is: \<N\>".
- `gsm8k-truncated-*`  — solution truncated to "The answer is" (no number).

Each family has three sized train variants (`-`, `-double`, `-all`) and
100 auto-discovered per-problem eval tasks (paired across families).

Build choices documented in `scripts/dataset_builder/GSM8K_BUILD_LOG.md`
(v1 section). Key points:

1. **100/639 problem split** via a new `--num-test-problems` flag in
   `build_gsm8k_dataset.py` — RLHFlow only labels GSM8K test, so the
   pre-existing "classify by question_id" mode was leaving train empty.
   The new flag pools all 739 qualified problems, deterministically holds
   out 100 (split-seed=42) for per-problem eval, puts the other 639 in
   `train.csv`. This is *not* an OOD-vs-GSM8K-train split — it's a
   sub-split of the same source pool — but it's a real OOD-by-problem
   split for our purposes.
2. **Paired full/truncated**: solutions are selected once per question
   and both versions write the same selections (truncated is exactly the
   prefix of full through "The answer is"). Earlier behaviour shuffled
   inside the version loop and produced different traces for the two
   versions, breaking the paired comparison.
3. **`ки` step-marker stripping**: ran `strip_step_markers.py` on the
   raw RLHFlow JSONL once to produce a clean intermediate, then point
   the builder at the clean file. The ки characters are PRM artifacts,
   not meaningful content.
4. **Sized variants match other modern families** (humaneval 2,744 /
   codecontests-base 2,522 / ifeval-concat 3,160). Small variant uses
   45 problems × ~56 rows ≈ 2,500 rows.
5. **Naming**: per-problem task names are `gsm8k-full-gsm8k_test_<N>`
   and `gsm8k-truncated-gsm8k_test_<N>` — the slug repeats the source
   `question_id`, mirroring humaneval's `humaneval-humaneval_<N>` pattern.

The clean JSONL (~1.2 GB) is regenerated locally and not committed.

---

## 2026-05-01

### EOS training: tried, didn't help

Trained variants with `--include-eos` so that `log P(EOS | prompt, completion)`
gets folded into the generator score during training (and matching `_eos`
evals via `eval_by_claude.py --include-eos`). The hypothesis was that
including EOS would discipline the generator's "where to stop" behavior
and tighten the gap. In practice the EOS-trained models did not show a
meaningful improvement over the non-EOS counterparts on the metrics we
care about, so EOS is not a default we want going forward.

Artifacts: models in `models-eos/`, eval scores in `outputs-eos-models/`.
The 18 hypernym tasks × 6 training settings × 3 base models matrix is fully
populated (see [`docs/eval_train_status.md`](eval_train_status.md)) so the
comparison is robust, not just under-evaluated.

### `basetyp` (combined with `neg` or `self`): probably the right way to do TC for fine-tuned models

The intuition: a fine-tuned model's own unconditional distribution P_FT(y)
is already shifted by the fine-tuning, so subtracting it as a typicality
correction at eval time understates how much the fine-tune actually moved
the conditional distribution. Using the **base (pre-finetune) model** as
the typicality prior — `--base-typicality` in `eval_by_claude.py`, with
`--base-model <name>` to pin which base — gives a correction that is
independent of how the model was fine-tuned and so is comparable across
training variants.

Two combinations matter:
- `basetyp-`  + `_tc`  →  P_base(y | null)            (self-style, base model)
- `basetypneg-` + `_tc` →  P_base(y | negated_prompt) (neg-style, base model)

This pairs cleanly with the existing self/neg eval prefixes and is the
"probably correct" default for evaluating fine-tuned models regardless of
their training-time TC choice.

Coverage: the basetyp / basetypneg hypernym evals are largely done for
2b, 2b-it, and 9b-it (see [`docs/eval_train_status.md`](eval_train_status.md)
workstream 2).

### Currently looking for: a task where our method beats vanilla RankAlign

The "vanilla RankAlign" baseline in our experiments is `pref-only` with no
NLL terms, no `force-same-x`, no typicality correction during training.
Our two main novelties on top of that are:

1. **Typicality-corrected generator scores during training** (frequency
   correction) — `--typicality-correction`, `--self-typicality`, or
   `--neg-typicality`.
2. **`--force-same-x` pair sampling** — pairs are drawn only from items
   that share the same generator prompt, so the pair contrasts genuinely
   different completions for the same input rather than mixing inputs.

Plus the optional NLL auxiliary terms (`--nll_validator_weight`,
`--nll_generator_weight`) and validator log-odds during training
(`--validator-log-odds`).

The empirical situation: it has been hard to find a task where this
combination decisively beats vanilla RankAlign. This is the open question
right now. We do not want to keep scaling experiments breadth-first
without finding at least one task where the novelties show a clear win.
Candidate tasks to keep probing here include codecontests and humaneval
(both newer, longer-context); it is also worth re-examining whether
`force-same-x` is actually doing what we think it is on the tasks where
it currently looks neutral.

### Make-it-easy-to-add-tasks refactor: in progress, mostly working

Recent commits set up the task registry, generic CSV row builder
callbacks, and the `tasks/common.py` helpers. The intent is that adding a
new task takes one self-contained module plus one import line in
`src/tasks/__init__.py`, with no edits to `ranking_loss_ref.py` or
`eval_by_claude.py` for non-legacy task families.

What works well today:
- A clean smoke test (`import tasks; len(TASK_REGISTRY)`) registers
  ~1000 tasks across 12 families with no errors.
- Auto-discovery patterns (drop a CSV/JSONL into `data/<family>/`)
  register new datasets with no Python edits.
- The generic CSV writer path (`csv_header` + `csv_row_builder` →
  `tasks/common.write_scores_csv`) means new families don't need new
  branches in `eval_by_claude.py`. codecontests and humaneval already use
  this path.
- [`docs/adding_new_data_and_tasks.md`](adding_new_data_and_tasks.md) is
  accurate and matches the code.

Known boilerplate / friction worth fixing (small, none blocking):

- The `sys.path` hack
  (`_parent_dir = os.path.dirname(os.path.dirname(...))` + `sys.path.insert`)
  is duplicated in every task module. Could be hoisted into
  `src/tasks/__init__.py` once.
- Adding a new task family still requires two steps: `register_task({...})`
  in the module and `from . import my_task` in `src/tasks/__init__.py`. A
  `pkgutil.iter_modules` autoloader would cut this to one step but at the
  cost of letting a single broken module break import for everyone — the
  current explicit-import behavior is probably fine.
- `register_task` silently overwrites duplicate names. A warning would
  catch accidental collisions cheaply.
- Legacy CSV-writing branches in `eval_by_claude.py` (hypernym, ifeval,
  ambigqa, plausibleqa, membership, rosch) and the parallel legacy
  `if/elif` chains in `ranking_loss_ref.py` still exist alongside the
  registry path. New tasks in those families inherit the legacy code
  paths even when their registry entry would let them use the generic
  one. Migrating these is the largest remaining cleanup; it's not
  required for adding new tasks but it would shrink both files
  significantly.

### Noble-gas / structural-r line: paused

The structural decomposition pipeline in [`docs/plan.md`](plan.md) (config
elicitation → π/π′ estimation → comparing log r̂(a) against the directly
observed residual s_V − (s_G − s_G′)) is on hold. We may come back to it
later, but it is not the active priority right now.

---

## Notes on `make_and_format_data` reproducibility (housekeeping)

`src/utils.py:make_and_format_data` calls `random.shuffle(items)` on the
global `random` module inside the `both='union'` and `both='joint'`
branches. It does not take a `seed` argument. In the current main
training pipeline (`ranking_loss_ref.py`, `ranking_loss_ref2.py`,
`fine_tune_lora.py`, `consistency_ft.py`) every call site passes
`both=None` and so falls into the `else` branch, which does not shuffle —
the unseeded shuffle never fires today. If we ever re-enable the union /
joint mixed-style training branches, the shuffle should take a local
`random.Random(seed)` to be properly reproducible. Not a current bug,
just a sharp edge.
