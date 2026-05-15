# May 2 vs May 13 cohort drift — investigation plan

## The puzzle

Two pref-only g-mode runs on `membership-sans-rosch-v0-all` with
`gemma-2-2b`, same `delta=0.15`, 3 epochs, that should be
**mathematically identical** (vlo and semi are both no-ops in this
configuration — see `docs/membership_to_rosch_recipe_inventory.md`)
produce gen-ROC numbers (mean × 100 across 10 rosch tasks) that
disagree:

| Should-be-identical pair | May 13 (no-vlo, no-semi) | May 2 (vlo+semi) | Δ |
| --- | --- | --- | --- |
| fsx + TC-self, self ref | 86.70 | 81.31 | **−5.39** |
| fsx + TC-neg, neg ref | 80.82 | 82.01 | +1.19 |

If vlo and semi are no-ops, the gap must be something else. We need
to find what.

## Phase 1 — Inventory the actual training invocations

### May 2 cohort: launcher chain

`scripts/run_train_membership.sh` (commit `24fefbf3`, 2026-05-02 18:16) ⇒
`scripts/run_train_semi.sh` ⇒ `python ranking_loss_ref.py ...`

Working directory at python time: `scripts/` (the wrapper does
`cd "$(dirname "$0")"`).

For the **fsx + TC-self + vlo (May 2)** cell the resolved python
command is:

```bash
python ranking_loss_ref.py \
    --model google/gemma-2-2b \
    --num_epochs 3 \
    --task membership-sans-rosch-v0 \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 0 \
    --nll_generator_weight 0 \
    --preference_loss_weight 1 \
    --all \
    --delta 0.15 \
    --force-same-x \
    --self-typicality \
    --validator-log-odds \
    --semi-supervised 0.1
```

(Run from `scripts/`, so `--models-dir` defaults to `../models`
which resolves to repo-root `models/`.)

**Argparse defaults that fill in the rest** (from
`scripts/ranking_loss_ref.py @ 24fefbf3`):

- `--learning_rate = 1e-5`
- `--total_samples = 5110`
- `--save_steps = 1`
- `--alpha = '1.0'`
- `--split-seed = 42`
- `--with_ref = False`
- `--length-normalize = False`
- `--include-eos = False`
- `--track-scores = False`
- `--no-upload-hf = False` (i.e. **uploads to HF Hub by default**)
- `--no-wandb = False`

### May 13 cohort: launcher chain

`scripts/run_train_membership_quickiter.sh` (commit on/around
2026-05-13) ⇒ `python scripts/ranking_loss_ref_online.py ...`

Working directory at python time: repo root (the wrapper does
`cd "$(dirname "$0")"/..`).

For the **fsx + TC-self (May 13)** cell the resolved python
command is:

```bash
python scripts/ranking_loss_ref_online.py \
    --model google/gemma-2-2b \
    --task membership-sans-rosch-v0 \
    --train_g_or_d g \
    --split_type random \
    --num_epochs 3 \
    --delta 0.15 \
    --save_steps 999 \
    --all \
    --force-same-x \
    --nll_validator_weight 0 \
    --nll_generator_weight 0 \
    --preference_loss_weight 1 \
    --models-dir ./models-quickiter \
    --no-upload-hf \
    --self-typicality
```

**Argparse defaults that fill in the rest** (from
`scripts/ranking_loss_ref_online.py @ HEAD`):

- `--learning_rate = 1e-5`
- `--total_samples = 5110`
- `--alpha = '1.0'`
- `--split-seed = 42`
- `--with_ref = False`
- `--length-normalize = False`
- `--include-eos = False`
- `--track-scores = False`
- `--no-wandb = False`
- `--online-pair-selection = False`
- `--online-typicality = False`

### Side-by-side: every flag, everywhere

| Flag / setting | May 2 (vlo+semi) | May 13 (no vlo, no semi) | Identical? |
| --- | --- | --- | --- |
| Python script | `scripts/ranking_loss_ref.py @ 24fefbf3` | `scripts/ranking_loss_ref_online.py @ HEAD` | ❌ different files, different commits |
| Working dir at python time | `scripts/` | repo root | ❌ different |
| `--model` | `google/gemma-2-2b` | `google/gemma-2-2b` | ✅ |
| `--task` | `membership-sans-rosch-v0` | `membership-sans-rosch-v0` | ✅ |
| `--num_epochs` | 3 | 3 | ✅ |
| `--learning_rate` | 1e-5 (default) | 1e-5 (default) | ✅ |
| `--delta` | 0.15 | 0.15 | ✅ |
| `--total_samples` | 5110 (default) | 5110 (default) | ✅ |
| `--save_steps` | 1 (default) | **999 (explicit)** | ❌ — investigate |
| `--all` | True (default) | True | ✅ |
| `--train_g_or_d` | g | g | ✅ |
| `--split_type` | random | random | ✅ |
| `--alpha` | 1.0 (default) | 1.0 (default) | ✅ |
| `--lora` | False (gemma-2-2b) | False (gemma-2-2b) | ✅ |
| `--gradient_checkpointing` | False | False | ✅ |
| `--typicality-correction` | False (only `--self-typicality`) | same | ✅ |
| `--self-typicality` | True | True | ✅ |
| `--neg-typicality` | False | False | ✅ |
| `--no-full-completion` | False | False | ✅ |
| `--single_token_data_only` | False | False | ✅ |
| `--preference_loss_weight` | 1 | 1 | ✅ |
| `--nll_validator_weight` | 0 | 0 | ✅ |
| `--nll_generator_weight` | 0 | 0 | ✅ |
| `--no-wandb` | False | False | ✅ |
| `--no-v2` | False | False | ✅ |
| **`--validator-log-odds`** | **True** | False | ❌ — but no-op (verified) |
| `--length-normalize` | False | False | ✅ |
| `--track-scores` | False | False | ✅ |
| `--force-same-x` | True | True | ✅ |
| `--boost-initial-val` | False | False | ✅ |
| **`--semi-supervised`** | **0.1** | None | ❌ — but no-op (verified) |
| `--labeled-only` | None | None | ✅ |
| `--split-seed` | 42 (default) | 42 (default) | ✅ |
| `--disc-shots` | None | None | ✅ |
| `--include-eos` | False | False | ✅ |
| `--models-dir` | `../models` (default, from `scripts/`) | `./models-quickiter` | ❌ — output path only, content semantically equiv |
| `--no-upload-hf` | False (i.e. uploads) | True (no upload) | ❌ — checkpoint on-disk should be the same |
| `--max-seq-len` | None (default) | None (default) | ✅ |
| `--online-pair-selection` | n/a (flag didn't exist) | False | n/a |
| `--online-typicality` | n/a (flag didn't exist) | False | n/a |

### What stands out at the flag level

After excluding `vlo` and `semi` (verified no-ops) and the
output-path / HF-upload diffs (cosmetic), the **only behavior-affecting
flag difference** is:

- **`--save_steps`**: 1 (May 2 default) vs 999 (May 13 explicit).

  In our training code `save_steps` controls "save model every N
  epochs". With `save_steps=1`, the May 2 run wrote a checkpoint
  after every epoch (so `epoch0`, `epoch1`, `epoch2` all exist on
  disk, plus a final). With `save_steps=999`, the May 13 run only
  wrote `epoch0` (initial) and `epoch2` (final), skipping `epoch1`.
  We need to verify whether the per-epoch save path includes an
  optimizer/RNG state save+restore that perturbs subsequent training
  steps. **Plausible but not yet confirmed.**

The remaining difference is **the script + commit**: May 2 ran
`ranking_loss_ref.py @ 24fefbf3`, May 13 ran
`ranking_loss_ref_online.py @ ~May 13 HEAD`. Even with identical
flags, these are different code. Phase 2 below diffs them.

## Phase 2 — Code-drift audit

### Phase 2a: Training-script drift

Diff `ranking_loss_ref.py @ 24fefbf3` vs `ranking_loss_ref_online.py @ HEAD`.

Steps:
1. `git log` files between the two commits to enumerate touching commits.
2. Get a real `git diff` between the two states.
3. Categorize each non-trivial hunk:
   - dataloader / pair-construction
   - typicality-score computation
   - loss math
   - optimizer / scheduler / step
   - tokenizer / prompt formatting
   - random-state handling
4. Rule each hunk in or out for the (g-mode + pref-only + self-TC + fsx)
   path that both runs exercise.

### Phase 2b: Eval-pipeline drift

The score CSVs were emitted by `scripts/eval_by_claude.py` (or
`eval.py`?) and then aggregated by `scripts/summarize_scores_file.py`.

Steps:
1. Identify exactly which eval entry-point produced each cohort's
   CSVs (look at filename conventions, log lines).
2. `git log` `scripts/eval_by_claude.py` and `scripts/eval.py` between
   May 2 and May 13.
3. `git log` `scripts/summarize_scores_file.py` between May 2 and
   May 13.
4. Diff each, categorize hunks, rule in/out the same way.

### Phase 2c: Task-definition drift

Did `src/tasks/membership.py` or `src/tasks/rosch.py` (or the
underlying data files) change between May 2 and May 13?

Steps:
1. `git log` for `src/tasks/membership.py`, `src/tasks/rosch*.py`,
   any `data/*membership*` or `data/*rosch*` files.
2. If anything changed, diff and classify whether it affects the
   set of items, prompts, or labels.

### Phase 2d: `utils.py` / shared helpers

Steps:
1. `git log` for `src/utils.py` and any helper modules imported by
   training and eval (e.g. `make_prompt`, `make_and_format_data`,
   typicality helpers).

## Phase 3 — Non-code sources of drift (if Phase 2 leaves residual)

Brainstormed list, to chase only after Phase 2 is complete:

1. **Random seed of pair sampling.** The `--split-seed=42` only
   controls labeled/unlabeled split for `--semi-supervised`. The
   *pair sampling* in
   `scripts/ranking_loss_ref.py` uses `random.sample(...)` after
   `random.seed(...)` — but only if some seed was set globally.
   Need to grep for `random.seed`, `np.random.seed`, `torch.manual_seed`
   and verify the pair-sampling order is reproducible.

2. **DataLoader shuffle seed.** If `DataLoader(shuffle=True)` is used
   with no `generator=` argument, the shuffle order is process-RNG
   dependent. Two runs from clean processes will see different batch
   orderings, which interacts with non-zero learning rate to produce
   different model states.

3. **PyTorch determinism.** `torch.use_deterministic_algorithms`,
   cuDNN benchmarking, `torch.cuda.manual_seed_all`. Default is
   non-deterministic. Two GPUs / two cuDNN versions / two driver
   versions can produce slightly different outputs even from the
   same seed.

4. **HuggingFace model revision pinning.** `from_pretrained(model_name)`
   without `revision=...` pulls "the current default branch tip" of
   the HF repo. If `google/gemma-2-2b` was updated between May 2
   and May 13, the two runs literally start from different weights.
   *(Probably unlikely for a Google base model, but worth a one-line
   check.)*

5. **Tokenizer version.** `transformers` upgrades sometimes change
   tokenizer behavior (e.g. how special tokens are added). If the
   conda env was upgraded between May 2 and May 13, prompts are
   tokenized differently → different scores → different pair
   selection → different training trajectory.

6. **CUDA / cuBLAS / cuDNN nondeterminism.** Even with identical
   code and seeds, fp16/bf16 matmul reductions on GPU can produce
   slightly different outputs across runs of the same machine
   *and* certainly across machines. Effects are usually tiny per
   step but compound over 5110 × 3 steps.

7. **Hardware difference.** May 2 and May 13 runs probably landed
   on different Slurm nodes (different GPU revisions, possibly
   different driver versions). Worth pulling Slurm logs to see
   which node each ran on.

8. **Floating-point reductions over batch order.** Batch-size 1 vs
   batch-size N changes reduction order. Both runs use the same
   default batch size, so this should be a no-confound — but worth
   verifying.

9. **Reference data on disk.** If `data/membership_*.json` or any
   prompt-template file changed between May 2 and May 13, prompts
   are different → all downstream is different.

10. **Wandb side effects.** Wandb's auto-instrumentation can call
    `torch.autograd.set_detect_anomaly(True)` or similar when it
    detects certain conditions, which can change numeric behavior
    in subtle ways. Probably negligible, but listed for
    completeness.

## Phase 4 — Definitive test (if Phase 2/3 leave the gap unexplained)

If we can't pin the gap on a known code change or RNG issue, run
**one** new training job with the **May 13 launcher** but pass
`--validator-log-odds --semi-supervised 0.1` (i.e. add the no-op
flags). If we recover something close to 86.70 (the May 13 number),
then the May 2 gap really is cohort drift not visible in the diff
(probably hardware/CUDA-level). If we recover ~81 (the May 2
number), then we've reproduced the gap and can bisect from there.

## Phase 5 — Document and decide

Update `docs/membership_to_rosch_recipe_inventory.md` with the
finding. If the gap is "real" cohort drift, the existing +5.07
within-May-13 TC effect is still the cleanest claim and we don't
have to discount it. The "fsx alone" cross-cohort delta should be
re-derived from a clean Plain RankAlign retrain on the May 13 commit.
