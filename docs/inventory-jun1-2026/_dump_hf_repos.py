#!/usr/bin/env python
"""Dump all rankalign model repos on latkes + TAUR-dev with privacy status.

Writes _raw/hf_repos.json. Reproducible source data for the training inventory.
Run with the tools venv:
    /home/jdr/racas-more/llm-consistency-raca/.tools-venv/bin/python _dump_hf_repos.py
"""
import json
import os
from pathlib import Path

from key_handler import KeyHandler
from huggingface_hub import HfApi

KeyHandler.set_env_key()
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
api = HfApi(token=tok)

OUT = Path(__file__).parent / "_raw" / "hf_repos.json"
result: dict[str, list[dict]] = {}

for org in ["latkes", "TAUR-dev"]:
    repos = []
    for m in api.list_models(author=org, limit=1000):
        name = m.id
        if "rankalign" not in name.lower() and "-v7-" not in name.lower():
            continue
        try:
            info = api.model_info(name)
            private = bool(info.private)
        except Exception as e:  # noqa: BLE001 - record the error, don't hide it
            private = None
            print(f"WARN model_info failed for {name}: {e}")
        repos.append({"name": name, "private": private})
    result[org] = sorted(repos, key=lambda r: r["name"])
    print(f"{org}: {len(repos)} rankalign/v7 model repos")

OUT.write_text(json.dumps(result, indent=2))
print(f"wrote {OUT}")
