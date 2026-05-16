# Pod orchestration scripts — gemma-4-31B-it humaneval-v2.1correct-upper TC run

These are the exact scripts that ran on RunPod pod `h100-train-4`
(`d4gum0mttfqkre`) in May 2026 to train + eval the priority-1 TC comparison
(RankAlign #2 vs RankAlign+tc #6 vs RankAlign+negtc #9) on
`humaneval-v2.1correct-upper` with `google/gemma-4-31B-it`.

Pulled into the repo so the run is reproducible and the training provenance is
durable (pod volumes are not). Paths inside the scripts are pod-absolute
(`/workspace/...`) — they are kept as a record of what ran, not as portable
scripts.

**One redaction:** the pod scripts had a hardcoded
`export HF_TOKEN=hf_...` (4 files). That literal token was replaced with
`export HF_TOKEN="${HF_TOKEN:?...}"` before committing — it is the only change
from the on-pod versions. The token was live on the pod and should be rotated.

## Which script did what (chronological)

| Script | Role |
|---|---|
| `sequential_pipeline.sh` | Earliest attempt — sequential train+eval for multiple models (gemma-2-27b-it, gemma-4-31B/-it). Superseded; the 27b-it logs from it are the "failed attempt" the user said to ignore. |
| `train_seq_v21cu.sh` | Helper invoked by the sequential pipeline to train one (model, TC) cell. |
| **`real_pipeline_g4it.sh`** | **The pipeline that trained the no_tc / RankAlign #2 baseline.** `COMMON` uses `--num_epochs 3 --total_samples 5110 --delta 0.15 --semi-supervised 0.1 --lora` (pref-only, no fsx). Trained `no_tc` → saved its **epoch0** adapter → then crashed at the LoRA-merge OOM before self_tc/neg_tc. |
| `merge_no_tc.py` | One-off attempt to merge the no_tc LoRA adapter into the base model. Abandoned — merge can't fit in the pod's 100 GB volume (62 GB base + 64 GB merged). Superseded by passing the PEFT adapter directly to `eval_by_claude.py`. |
| **`recovery_pipeline_v2.sh`** | **The recovery run.** Reused the existing no_tc epoch0 adapter (hard `exit 1` if missing — never retrains it), trained `self_tc`/`neg_tc` fresh with the same `COMMON` **except `--num_epochs 1`**, merge disabled, PEFT adapter passed directly to eval. Produced the 328 score files. |
| `recovery_pipeline.sh` | First (v1) recovery attempt, superseded by v2. |
| `watch_recovery.sh` | Pod-side poller that watched `recovery_v2.log` for `VARIANT_*_COMPLETE` markers. |

## Why the #2-vs-#6/#9 comparison is still clean despite num_epochs 3 vs 1

The baseline ran `--num_epochs 3` but **we compare its `epoch0` checkpoint**,
i.e. the model after exactly **one** pass over the 5110-sample set. The TC
models ran `--num_epochs 1`; we also compare their `epoch0` checkpoint — also
after one pass. `--num_epochs` only bounds the training loop
(`for epoch in range(num_epochs)`) and the save condition; it does **not**
change anything that happens during epoch 0. `ranking_loss_ref_gemma4.py` has
**no LR scheduler** (`AdamW(model.parameters(), lr=lr)`, constant lr=1e-5, no
`scheduler.step()`), so the optimizer trajectory through epoch 0 is identical
regardless of total scheduled epochs. All three compared checkpoints = same
1-epoch training budget; only the TC training flag differs.

Full provenance + results:
`notes/log_P_diff_plots/humaneval-v2.1correct-upper/trained_eval_outputs/_analysis/PROVENANCE.md`.
