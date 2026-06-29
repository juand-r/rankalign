#!/usr/bin/env python3
"""Upload one qwen IFEval RankAlign(s2) variance-run checkpoint to latkes (PUBLIC).

Pod-side. After run_qwen35_variance_overnight.sh finishes, this uploads the epoch2 merged
model + LoRA adapter + training logs to a public latkes repo so the trained model is backed
up off the pod BEFORE the pod is stopped.

Usage on the pod:
    HF_TOKEN=hf_... python upload_qwen_variance_to_latkes.py 1

Repo: latkes/rankalign-v7-qwen3.5-9b-ifeval-s2-variance-run{RUN_ID}
"""
import glob
import os
import sys
import textwrap

from huggingface_hub import HfApi

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_ID", "")
assert RUN_ID, "RUN_ID required (1|2|3)"
TOKEN = os.environ["HF_TOKEN"]
REPO = f"latkes/rankalign-v7-qwen3.5-9b-ifeval-s2-variance-run{RUN_ID}"

MODELS_DIR = "/workspace/models_q35"
glob_pat = (f"{MODELS_DIR}/v7-Qwen--Qwen3.5-9B-delta*-epoch2--ifeval-concat-all--d2g--random--"
            f"alpha1.0--full-completion--vallogodds--semi0.1--fix1_merged")
merged = sorted(glob.glob(glob_pat))
assert merged, f"no epoch2 merged dir matching {glob_pat}"
merged_dir = merged[-1]
delta_tag = os.path.basename(merged_dir).split("-delta")[1].split("-epoch")[0]
print(f"merged dir: {merged_dir}  (delta tag {delta_tag})")

api = HfApi(token=TOKEN)
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)

readme = textwrap.dedent(f"""\
    ---
    base_model: Qwen/Qwen3.5-9B
    library_name: transformers
    tags: [rankalign, qwen3.5-9b, lora-merged, variance-test]
    ---
    # RankAlign — Qwen3.5-9B — IFEval — variance run {RUN_ID}

    Merged (base + LoRA) Qwen3.5-9B RankAlign (setting s2) model, trained on a fresh RunPod pod
    to measure run-to-run variance vs the original 83.2 checkpoint
    (`latkes/rankalign-v7-qwen3.5-9b-ifeval-s2-ep2`).

    **Recipe (identical to the original):** `run_qwen35_cell.sh ifeval s2` —
    `ranking_loss_ref_fix.py --task ifeval-concat --train_g_or_d g --all --delta 0.15
    --delta-bins 10 --preference_loss_weight 1 --semi-supervised 0.1 --validator-log-odds
    --lora --num_epochs 3`. Auto-delta -> {delta_tag}.

    **Environment (identical to the original):** RunPod `runpod/pytorch:2.4.0` image
    (torch 2.4.1), `requirements-gemma4.txt` (transformers 5.8.1), Qwen3.5 torch-fallback
    attention.

    Differs from the original only by the unseeded pair shuffle + GPU nondeterminism — this is
    run {RUN_ID} of 3. LoRA adapter under `adapter/`, training log `cell_ifeval_s2.log`.
    """)
with open("/tmp/README.md", "w") as f:
    f.write(readme)

print(f"uploading merged model dir -> {REPO} (public) ...")
api.upload_folder(folder_path=merged_dir, repo_id=REPO, repo_type="model",
                  commit_message=f"variance run {RUN_ID}: epoch2 merged model (delta {delta_tag})")
api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md",
                repo_id=REPO, repo_type="model")
for logf in glob.glob("/workspace/logs/cell_ifeval_s2.log") + glob.glob(f"/workspace/logs/variance_run{RUN_ID}.log"):
    api.upload_file(path_or_fileobj=logf, path_in_repo=f"logs/{os.path.basename(logf)}",
                    repo_id=REPO, repo_type="model")

print(f"DONE: https://huggingface.co/{REPO}")
