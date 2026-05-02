# Updates / Status (2026-05-01)

A running, dated log of where things stand on the methods side and what's
worth keeping in mind when planning next steps. Append new dated sections
on top; don't rewrite history.

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
