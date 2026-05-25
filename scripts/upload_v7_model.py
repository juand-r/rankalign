"""Upload a v7 rankalign model checkpoint to HuggingFace.

Simple alternative to upload_checkpoint.py that handles v7 naming convention.
Derives a short, readable repo name from the checkpoint dirname.

Usage:
    HF_TOKEN=hf_... python upload_v7_model.py --local-path /path/to/model_merged --hf-org TAUR-dev
"""

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def derive_repo_name(local_path: str) -> str:
    """Derive a short HF repo name from a v7 checkpoint directory name.

    e.g. v7-google--gemma-2-9b-it-delta2.49-epoch2--persona-v1-all--d2g--random--
         alpha1.0--full-completion--vallogodds--semi0.1--fix1_merged
    -> rankalign-v7-gemma2-9b-it-persona-v1-s2-epoch2

    Key fields extracted:
      - model short name (gemma2-9b-it or Qwen3.5-9B)
      - task (persona-v1 / membership-sans-rosch-v0 / ifeval-concat)
      - setting tag (derived from flags)
      - epoch
    """
    name = Path(local_path).name.replace("_merged", "")

    # Model
    if "gemma-2-9b-it" in name or "gemma--2-9b-it" in name:
        model_tag = "gemma2-9b-it"
    elif "Qwen3.5-9B" in name or "Qwen--Qwen3.5-9B" in name:
        model_tag = "qwen3.5-9b"
    else:
        model_tag = re.sub(r"[^a-zA-Z0-9-]", "-", name.split("-delta")[0].replace("v7-", ""))[:20]

    # Epoch
    em = re.search(r"epoch(\d)", name)
    epoch_tag = f"ep{em.group(1)}" if em else "ep?"

    # Task
    if "persona-v1-all" in name:
        task_tag = "persona"
    elif "membership-sans-rosch" in name:
        task_tag = "membership"
    elif "ifeval-concat" in name:
        task_tag = "ifeval"
    else:
        task_tag = "unknown"

    # Setting: derive from flags
    has_labelonly = "labelonly0.1" in name
    has_semi = "semi0.1" in name
    has_fsx = "force-same-x" in name
    has_tc_self = "tc-self" in name
    has_tc_neg = "tc-neg" in name
    has_nll = "nllv1.0" in name or "nllv1" in name

    if has_labelonly and not has_fsx:
        setting_tag = "s1"
    elif has_semi and not has_fsx and not has_nll:
        setting_tag = "s2"
    elif has_semi and has_fsx and not has_tc_self and not has_tc_neg:
        setting_tag = "s3"
    elif has_semi and has_fsx and has_tc_self:
        setting_tag = "s4"
    elif has_semi and has_fsx and has_tc_neg:
        setting_tag = "s7"
    else:
        setting_tag = "sX"

    repo_name = f"rankalign-v7-{model_tag}-{task_tag}-{setting_tag}-{epoch_tag}"
    # HF repo names: only alphanumeric, hyphens, underscores, dots; max 96 chars
    repo_name = re.sub(r"[^a-zA-Z0-9._-]", "-", repo_name)[:96]
    return repo_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--hf-org", default="TAUR-dev")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try huggingface cached token
        token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        sys.exit("ERROR: No HF_TOKEN found in env or ~/.cache/huggingface/token")

    local_path = Path(args.local_path)
    if not local_path.is_dir():
        sys.exit(f"ERROR: local path does not exist: {local_path}")

    repo_name = derive_repo_name(str(local_path))
    repo_id = f"{args.hf_org}/{repo_name}"

    print(f"[upload_v7_model] local_path : {local_path}")
    print(f"[upload_v7_model] repo_id    : {repo_id}")

    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"[upload_v7_model] repo exists, will upload/update")
    except Exception:
        print(f"[upload_v7_model] creating repo {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", private=True, token=token)

    print(f"[upload_v7_model] uploading {local_path} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload rankalign v7 checkpoint: {local_path.name}",
    )
    print(f"[upload_v7_model] DONE: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
