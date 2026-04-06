# HuggingFace Checkpoint Upload Plan

## Goal

Auto-upload model checkpoints to HuggingFace after each epoch save in `ranking_loss_ref.py`,
so that:
- Checkpoints are safe even if the cluster job dies mid-run
- Eval jobs (`eval_by_claude.py`) can load directly from HF via `--model juand-r/<repo-name>`
- All checkpoints are tracked in one place

---

## Files to Create

### 1. `src/checkpoint_name_parser.py` (new)

A pure utility module with no side effects. Two functions:

**`parse_checkpoint_name(folder_name: str) -> dict`**

Parses the long folder name produced by `ranking_loss_ref.py` into structured fields.

| Field | Example | Notes |
|-------|---------|-------|
| `version` | `v6` | Always present |
| `model_name` | `google/gemma-2-2b` | Reconstructed from `google--gemma-2-2b` |
| `model_short` | `gemma-2-2b` | Drops org prefix, used in HF repo name |
| `delta` | `0.15` | Float |
| `epoch` | `9` | Int |
| `task` | `hypernym-hammers-all` | Everything between epoch and direction flags |
| `tc` | `online` / `self` / `neg` / `None` | Typicality correction mode |
| `lenorm` | `True` / `False` | |
| `pref` | `1.0` / `None` | Preference loss weight, None if absent |
| `nll_v` | `1.0` / `None` | NLL validator weight, None if absent |
| `nll_g` | `1.0` / `None` | NLL generator weight, None if absent |
| `vallogodds` | `True` / `False` | |
| `force_same_x` | `True` / `False` | |
| `semi` | `0.1` / `None` | Semi-supervised ratio |
| `labelonly` | `0.1` / `None` | Labeled-only ratio |
| `original_name` | (full string) | Stored verbatim for reproducibility |

Fields that are **dropped** (not included in HF name): `d2g/g2d/iter/both`, `random/hyper/both`
(split type), `alpha{x}`, `full-completion`, `with-ref`, `all`, `single-token-data`, `valboost`.

**`to_hf_repo_name(parsed: dict, prefix: str = "rankalign") -> str`**

Builds the HF repo name from parsed fields. Only includes fields that are present/True.

Format:
```
{prefix}-{version}-{model_short}-delta{delta}-epoch{epoch}-{task}[-tc-{tc}][-lenorm][-pref{x}][-nllv{x}][-nllg{x}][-vallogodds][-force-same-x][-semi{x}][-labelonly{x}]
```

Examples:
```
rankalign-v6-gemma-2-2b-delta0.15-epoch9-hypernym-hammers-all-tc-online-nllv1.0-nllg1.0-vallogodds
rankalign-v6-gemma-2-2b-delta0.15-epoch2-plausibleqa-all-tc-self-lenorm-pref0.0-nllv1.0-nllg1.0-force-same-x-semi0.1
```

---

### 2. `src/upload_checkpoint.py` (new, standalone script)

Called as a **subprocess** from `ranking_loss_ref.py` so it never shares memory with the
training process. Uses `.tools-venv/bin/python` (has `huggingface_hub` and `key_handler`).

**Arguments:**
- `--local-path` — absolute path to checkpoint dir to upload (the `_merged` dir for LoRA, base dir for full fine-tuning)
- `--hf-org` — HF org (passed in as `juand-r`)
- `--experiment-notes-dir` — absolute path to the experiment's notes folder for updating `HUGGINGFACE_REPOS.md` (optional)

**What it does:**
1. Calls `KeyHandler.set_env_key()` to inject the HF token
2. Parses the local path's folder name via `parse_checkpoint_name()`
3. Builds the HF repo name via `to_hf_repo_name()`
4. Uploads via `huggingface_hub.upload_folder()` to `{hf_org}/{repo_name}` as a **model** repo
5. Generates a model card with:
   - All parsed metadata fields
   - `original_name` for exact reproducibility
   - Base model tag
6. Appends an entry to `HUGGINGFACE_REPOS.md` if `--experiment-notes-dir` is provided

**Does NOT use `hf_utility.push_dataset_to_hub()`** — that's for datasets. Model repos use
`huggingface_hub` directly.

---

## Files to Modify

### 3. `scripts/ranking_loss_ref.py` (small addition)

Add an `--upload-hf` flag (off by default, so existing runs are unaffected).

After the save block (currently ends at line 2684), add:

```python
if args.upload_hf:
    upload_path = merge_dir if use_lora else save_directory
    subprocess.Popen([
        "/datastor1/jdr/gv-gap/rankalign/../.tools-venv/bin/python",  # resolved at runtime
        str(Path(__file__).parent.parent / "src" / "upload_checkpoint.py"),
        "--local-path", upload_path,
        "--hf-org", "juand-r",
        "--experiment-notes-dir", args.experiment_notes_dir or "",
    ])
```

- `Popen` (non-blocking) — training continues immediately, upload happens in background
- HF org hardcoded to `juand-r` here since `.raca/config.yaml` is not on the cluster; alternatively passed via `--hf-org` arg
- `--experiment-notes-dir` is optional; if not passed, `HUGGINGFACE_REPOS.md` is just skipped

**New args added to `ranking_loss_ref.py`:**
- `--upload-hf` (flag, default False) — enable HF upload after each checkpoint save
- `--hf-org` (str, default `"juand-r"`) — HF org to upload to
- `--experiment-notes-dir` (str, default `""`) — path to notes dir for tracking

---

## What Stays the Same

- `eval_by_claude.py` — **no changes**. Already accepts HF model IDs via `--model`.
  To use a HF checkpoint: `--model juand-r/rankalign-v6-gemma-2-2b-delta0.15-epoch9-...`
- Existing training runs without `--upload-hf` are completely unaffected
- Local checkpoint saves are unchanged — HF upload is purely additive

---

## Notes

### HF org
**Decision: B** — pass `--hf-org juand-r` explicitly in the sbatch/shell script that calls `ranking_loss_ref.py`. No magic, no hardcoding in Python.

---

### Filename Parsing — Read These Files First

**Before writing `checkpoint_name_parser.py`, carefully read and follow the parsing logic in:**
- `scripts/dashboard_viz_refactor.py` — `parse_filename()` starting at line 149
- `scripts/dashboard_viz_refactor-semi.py` — `parse_filename()` starting at line 155
- `scripts/ranking_loss_ref.py` — the `save_directory` construction at line 2658 and the string-building variables at lines 2618–2658

These are the authoritative sources. Do not guess at the format — derive the parser directly from them.

**Key facts from the source:**

The `save_directory` is built as:
```
../models/v6-{model//--}-delta{delta}-epoch{epoch}--{task}{with_ref_str}{all_str}{direction_str}{split_type_str}{alpha_str}{typcorr_str}{lenorm_str}{single_token_str}{full_completion_str}{pref_str}{nll_v_str}{nll_g_str}{force_same_x_str}{valboost_str}{vallogodds_str}{semi_str}
```

**Exact flag strings produced by `ranking_loss_ref.py`:**

| Variable | Value when present | Value when absent |
|---|---|---|
| `with_ref_str` | `--with-ref` | `""` |
| `all_str` | `-all` | `""` ← single dash! |
| `direction_str` | `--g2d` / `--d2g` / `--iter` / `--both` | always present |
| `split_type_str` | `--random` / `--hyper` / `--both` | always present |
| `alpha_str` | `--alpha{x}` or `--alpha-{name}` | always present |
| `typcorr_str` | `--tc-neg` / `--tc-self` / `--tc-online` | `""` |
| `lenorm_str` | `--lenorm` | `""` |
| `single_token_str` | `--single-token-data` | `""` |
| `full_completion_str` | `--full-completion` | `""` |
| `pref_str` | `--pref{weight}` | `""` when weight == 1.0 |
| `nll_v_str` | `--nllv{weight}` | `""` when weight == 0.0 |
| `nll_g_str` | `--nllg{weight}` | `""` when weight == 0.0 |
| `force_same_x_str` | `--force-same-x` | `""` |
| `valboost_str` | `--valboost` | `""` |
| `vallogodds_str` | `--vallogodds` | `""` |
| `semi_str` | `--semi{ratio}` or `--labelonly{ratio}` | `""` |

**Critical defaults (from `dashboard_viz_refactor.py` lines 236–241):**
- `pref` is **omitted** from the filename when its value is `1.0` → when absent, effective value = `1.0`
- `nllv`/`nllg` are **omitted** when their value is `0.0` → when absent, effective value = `0.0`
- The parser must apply these defaults, not treat absence as `None`

**Parsing strategy:**
- Strip the `../models/` prefix, then the `v6-` prefix
- Extract model name: everything up to `-delta` (replace `--` back to `/`)
- Extract `delta`: float after `-delta`
- Extract `epoch`: int after `-epoch`
- Extract `task`: text between `-epoch{n}--` and the next `--` separator
- All remaining flags are `--`-separated tokens (except `all_str` which uses single `-`)
- Use regex matching identical to `dashboard_viz_refactor.py`'s `_extract_float()` for weights
