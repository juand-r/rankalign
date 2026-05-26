#!/usr/bin/env python
"""Upload a merged Qwen3.5-9B RankAlign model (+ adapter + training log + card) to HF.

Usage:
    upload_qwen_model_to_hf.py REPO MERGED_DIR ADAPTER_DIR LOG_FILE CARD_KEY

Runs on the RunPod pod where the merged weights live. Uploads:
  - merged model weights (upload_folder)
  - LoRA adapter (adapter_config.json + adapter_model.safetensors) under adapter/
  - gzipped training log under training_log.log.gz
  - a generated README.md model card (uploaded last so it isn't clobbered)
"""
import gzip
import os
import shutil
import sys

from huggingface_hub import HfApi

CARDS = {
    "ifeval-s1": """---
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags: [rankalign, qwen3.5-9b, ifeval, lora-merged]
---
# RankAlign v7 — Qwen3.5-9B — ifeval — setting s1 (SFT label-only), epoch 2

Merged (base + LoRA) Qwen3.5-9B fine-tuned on **ifeval-concat** with RankAlign **setting s1**
(SFT label-only baseline: no preference/ranking loss). Final checkpoint of a 3-epoch run (ep2).
delta = 0.84 (delta-bins scheme, 10 bins).

- Base model: `Qwen/Qwen3.5-9B`
- LoRA: r=16, alpha=32, dropout=0.1, targets q/k/v/o/gate/up/down_proj (see `adapter/`)
- Trained with `scripts/run_qwen35_cell.sh ifeval s1` (no-upload-hf, no-wandb)
- Eval (held-out ifeval test prompts, n=20, NO_BASE): gen_roc Raw 52.0 / TC(self) 73.1; val_roc 57.5; val_acc 44.7

Full provenance, logs, and reproduction notes:
`private_projects/rankalign/docs/qwen_model_uploads_2026-05-26/` in the rankalign repo.
""",
    "ifeval-s13": """---
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags: [rankalign, qwen3.5-9b, ifeval, consistency-ft, lora-merged]
---
# RankAlign v7 — Qwen3.5-9B — ifeval — setting s13 (SFT + consistency-FT), epoch 0

Merged (base + LoRA) Qwen3.5-9B fine-tuned on **ifeval-concat** with RankAlign **setting s13**
(SFT label-only + consistency fine-tuning / CFT). Trained for **1 epoch** (ep0 = final).
delta = 0.15 (fixed).

- Base model: `Qwen/Qwen3.5-9B`
- LoRA: r=16, alpha=32, dropout=0.1, targets q/k/v/o/gate/up/down_proj (see `adapter/`)
- Trained with `EPOCHS=1 scripts/run_qwen35_v7b_cell.sh ifeval s13` (no-upload-hf, no-wandb)
- Eval (held-out ifeval test prompts, n=20, NO_BASE): gen_roc Raw 52.5 / TC(self) 76.3 / TC(neg) 50.3; val_roc 54.1; val_acc 48.3

Full provenance, logs, and reproduction notes:
`private_projects/rankalign/docs/qwen_model_uploads_2026-05-26/` in the rankalign repo.
""",
    "membership-s13": """---
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags: [rankalign, qwen3.5-9b, membership, consistency-ft, lora-merged]
---
# RankAlign v7 — Qwen3.5-9B — membership — setting s13 (SFT + consistency-FT), epoch 0

Merged (base + LoRA) Qwen3.5-9B fine-tuned on **membership-sans-rosch-v0** with RankAlign
**setting s13** (SFT label-only + consistency fine-tuning / CFT). Trained for **1 epoch**
(ep0 = final). delta = 0.15 (fixed). Evaluated on the held-out **rosch** membership categories.

- Base model: `Qwen/Qwen3.5-9B`
- LoRA: r=16, alpha=32, dropout=0.1, targets q/k/v/o/gate/up/down_proj (see `adapter/`)
- Trained with `EPOCHS=1 scripts/run_qwen35_v7b_cell.sh membership s13` (no-upload-hf, no-wandb)
- Eval (rosch, 10 categories, n=10, NO_BASE): gen_roc Raw 77.7 / TC(self) 80.1 / TC(neg) 86.1; val_roc 95.0; val_acc 84.3

Full provenance, logs, and reproduction notes:
`private_projects/rankalign/docs/qwen_model_uploads_2026-05-26/` in the rankalign repo.
""",
}


def main() -> None:
    repo, merged_dir, adapter_dir, log_file, card_key = sys.argv[1:6]
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    print(f"[upload] repo ready: {repo}", flush=True)

    print(f"[upload] uploading merged weights from {merged_dir} ...", flush=True)
    api.upload_folder(folder_path=merged_dir, repo_id=repo, repo_type="model",
                      ignore_patterns=["*.log", "README.md"])
    print("[upload] merged weights done", flush=True)

    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        src = os.path.join(adapter_dir, fn)
        if os.path.exists(src):
            api.upload_file(path_or_fileobj=src, path_in_repo=f"adapter/{fn}",
                            repo_id=repo, repo_type="model")
            print(f"[upload] adapter/{fn} done", flush=True)

    if os.path.exists(log_file):
        gz = "/tmp/_train_log.log.gz"
        with open(log_file, "rb") as f, gzip.open(gz, "wb") as g:
            shutil.copyfileobj(f, g)
        api.upload_file(path_or_fileobj=gz, path_in_repo="training_log.log.gz",
                        repo_id=repo, repo_type="model")
        print("[upload] training_log.log.gz done", flush=True)

    card = CARDS[card_key]
    with open("/tmp/_card.md", "w") as f:
        f.write(card)
    api.upload_file(path_or_fileobj="/tmp/_card.md", path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    print(f"[upload] README.md done — ALL DONE: {repo}", flush=True)


if __name__ == "__main__":
    main()
