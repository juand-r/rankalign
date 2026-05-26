#!/usr/bin/env python3
"""Upload v7 correct-upper LoRA adapters to HuggingFace TAUR-dev.

Usage (on pod):
    HF_TOKEN=<token> python3 upload_cu_adapters.py

Uploads each v7-* folder in /workspace/models_g4it/ as a separate model repo.
Naming: rankalign-v7-g4-31b-{variant}-e{N}-cu-all
"""
import os
import sys
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_ORG = "latkes"
MODELS_DIR = Path("/workspace/models_g4it")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set", flush=True)
    sys.exit(1)

api = HfApi(token=HF_TOKEN)


def folder_to_repo_name(folder: str) -> str:
    """Convert long adapter folder name to short HF repo name."""
    # Extract key fields
    epoch_m = re.search(r"epoch(\d+)", folder)
    epoch = epoch_m.group(1) if epoch_m else "?"
    delta_m = re.search(r"delta([\d.]+)", folder)
    delta = delta_m.group(1) if delta_m else "?"

    flags = []
    if "--tc-self" in folder:
        flags.append("tcs")
    elif "--tc-neg" in folder:
        flags.append("tcn")

    if "--nllv1.0" in folder:
        flags.append("nv1")
    if "--nllg1.0" in folder:
        flags.append("ng1")
    if "--vallogodds" in folder:
        flags.append("vlo")
    if "--force-same-x" in folder:
        flags.append("fsx")
    if "--ppd" in folder:
        flags.append("ppd")
    if "--semi0.1" in folder:
        flags.append("sm0.1")
    if "--fix1" in folder:
        flags.append("fix1")

    flag_str = "-".join(flags)
    d_str = delta.replace(".", "")
    name = f"rankalign-v7-g4-31b-d{d_str}-e{epoch}-cu-all"
    if flag_str:
        name += f"-{flag_str}"
    return name


adapter_dirs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v7")])

if not adapter_dirs:
    print("No v7 adapter dirs found.", flush=True)
    sys.exit(0)

print(f"Found {len(adapter_dirs)} v7 adapter dirs:", flush=True)
for d in adapter_dirs:
    print(f"  {d.name}", flush=True)

uploaded = []
for adapter_dir in adapter_dirs:
    repo_name = folder_to_repo_name(adapter_dir.name)
    repo_id = f"{HF_ORG}/{repo_name}"
    print(f"\nUploading {adapter_dir.name}", flush=True)
    print(f"  → {repo_id}", flush=True)

    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN, private=True)
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {adapter_dir.name}",
        )
        print(f"  ✓ Done", flush=True)
        uploaded.append(repo_id)
    except Exception as e:
        print(f"  ✗ FAILED: {e}", flush=True)

print(f"\n=== Summary ===")
print(f"Uploaded {len(uploaded)}/{len(adapter_dirs)} repos:")
for r in uploaded:
    print(f"  https://huggingface.co/{r}")
