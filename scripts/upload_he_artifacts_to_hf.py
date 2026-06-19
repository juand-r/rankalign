"""Upload humaneval-rerun artifacts (scores + gemma-4 adapters + qwen-3.5 merged models)
to the latkes HF org (PUBLIC). Runs ON mll. Resumable + idempotent.

Token: read from /datastor2/jdr/.hf_token (mode 600, NOT committed). Never hardcoded.

Repos (all latkes, public):
  - latkes/humaneval-rerun-scores            (dataset)  -- humaneval score CSVs
  - latkes/humaneval-rerun-gemma4-adapters   (model)    -- new gemma-4 s1/s3/s7/s13 LoRA adapters
  - latkes/humaneval-rerun-qwen35-merged     (model)    -- all 36 qwen-3.5 merged checkpoints

Each source dir is uploaded as a subfolder (path_in_repo = dir basename) so one repo holds
the whole family. Per-dir done-markers under $REPO/.he_monitor/hf_upload_done/ give coarse
resumability; huggingface_hub also skips already-uploaded files within a dir on retry.

Usage (on mll, qwen35 venv active, HF_TOKEN exported):
  python upload_he_artifacts_to_hf.py --target scores
  python upload_he_artifacts_to_hf.py --target adapters
  python upload_he_artifacts_to_hf.py --target qwen [--limit 1]   # --limit 1 = canary one dir
  python upload_he_artifacts_to_hf.py --target all
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path("/datastor2/jdr/rankalign")
DONE = REPO / ".he_monitor" / "hf_upload_done"
ORG = "latkes"
SCORES_REPO = f"{ORG}/humaneval-rerun-scores"
ADAPTERS_REPO = f"{ORG}/humaneval-rerun-gemma4-adapters"
QWEN_REPO = f"{ORG}/humaneval-rerun-qwen35-merged"

# score source dirs -> subfolder name in the dataset repo (humaneval files only)
SCORE_DIRS = {
    "outputs-he-trainset": "train-scores",
    "outputs-rerun-wandb": "qwen-test-scores",
    "outputs_gemma4_mll_tmp": "gemma-upper-test-scores",
    "outputs_gemma4_mll_tmp-multi": "gemma-multi-test-scores",
}


def token() -> str:
    t = os.environ.get("HF_TOKEN")
    if not t:
        # token lives OUTSIDE the git repo (parent dir) so it can never be committed
        t = Path("/datastor2/jdr/.hf_token").read_text().strip()
    if not t:
        sys.exit("ERROR: no HF token (env HF_TOKEN or /datastor2/jdr/.hf_token)")
    return t


def marker(name: str) -> Path:
    DONE.mkdir(parents=True, exist_ok=True)
    return DONE / f"{name}.done"


def upload_one_folder(api, repo_id, repo_type, folder: Path, path_in_repo: str, tag: str,
                      allow_patterns=None, ignore_patterns=None):
    m = marker(f"{repo_type}_{tag}")
    if m.exists():
        print(f"  [skip] {tag} (done-marker present)", flush=True)
        return
    print(f"  [upload] {folder}  ->  {repo_id}:{path_in_repo}", flush=True)
    api.upload_folder(
        repo_id=repo_id, repo_type=repo_type, folder_path=str(folder),
        path_in_repo=path_in_repo, allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=f"Add {path_in_repo}",
    )
    m.write_text("ok\n")
    print(f"  [done]  {tag}", flush=True)


def do_scores(api):
    api.create_repo(SCORES_REPO, repo_type="dataset", private=False, exist_ok=True)
    for src, sub in SCORE_DIRS.items():
        d = REPO / src
        if not d.exists():
            print(f"  [warn] missing {d}"); continue
        # only the score CSVs; exclude the eval run's .done marker dirs (they also
        # contain "humaneval" in their names and would otherwise pollute the repo)
        upload_one_folder(api, SCORES_REPO, "dataset", d, sub, f"scores_{sub}",
                          allow_patterns=["*humaneval*.csv"],
                          ignore_patterns=[".done/*", "*/.done/*", "*.done"])


def do_adapters(api):
    api.create_repo(ADAPTERS_REPO, repo_type="model", private=False, exist_ok=True)
    for src, sub in [("gemma-4-models-mll-tmp", "upper"),
                     ("gemma-4-models-mll-tmp-multi", "multi")]:
        d = REPO / src
        if not d.exists():
            print(f"  [warn] missing {d}"); continue
        # only the gemma-4 v7 checkpoint dirs (skip stray files)
        for ckpt in sorted(d.glob("v7-*gemma-4*")):
            if ckpt.is_dir():
                upload_one_folder(api, ADAPTERS_REPO, "model", ckpt,
                                  f"{sub}/{ckpt.name}", f"adapter_{sub}_{ckpt.name}")


def do_qwen(api, limit=None):
    api.create_repo(QWEN_REPO, repo_type="model", private=False, exist_ok=True)
    src = REPO / "models2-rerun-wandb"
    dirs = sorted(p for p in src.glob("v7-Qwen--Qwen3.5-9B-*humaneval*_merged") if p.is_dir())
    print(f"  qwen merged dirs found: {len(dirs)}", flush=True)
    if limit:
        dirs = dirs[:limit]
        print(f"  --limit {limit}: uploading first {len(dirs)} only (canary)", flush=True)
    for ckpt in dirs:
        upload_one_folder(api, QWEN_REPO, "model", ckpt, ckpt.name, f"qwen_{ckpt.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["scores", "adapters", "qwen", "all"])
    ap.add_argument("--limit", type=int, default=None, help="cap number of qwen dirs (canary)")
    args = ap.parse_args()
    api = HfApi(token=token())
    me = api.whoami()
    assert ORG in [o["name"] for o in me.get("orgs", [])], f"token cannot write to {ORG}"
    print(f"authenticated as {me['name']}, {ORG} writable. target={args.target}", flush=True)
    if args.target in ("scores", "all"):
        print("=== SCORES ==="); do_scores(api)
    if args.target in ("adapters", "all"):
        print("=== GEMMA-4 ADAPTERS ==="); do_adapters(api)
    if args.target in ("qwen", "all"):
        print("=== QWEN-3.5 MERGED ==="); do_qwen(api, limit=args.limit)
    print("ALL REQUESTED UPLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
