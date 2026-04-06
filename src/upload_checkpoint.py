"""
Upload a rankalign model checkpoint to HuggingFace Hub.

Called as a subprocess from ranking_loss_ref.py after each epoch save.
Runs with the same Python/venv as the training job (transformers is available,
so huggingface_hub is available). Token is read from HF_TOKEN env var or
~/.huggingface/token (set via `huggingface-cli login` on the cluster).

Usage:
  python src/upload_checkpoint.py \\
    --local-path /path/to/checkpoint \\
    --hf-org juand-r \\
    [--experiment-notes-dir /path/to/notes/experiments/my-exp]
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_name_parser import parse_checkpoint_name, to_hf_repo_name

from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO, format='[upload_checkpoint] %(message)s')
log = logging.getLogger(__name__)


# Maps concat training task names to individual eval subtask names.
# Add new entries here when new concat tasks are introduced.
CONCAT_TASK_SUBTASKS = {
    'hypernym-concat-bananas-to-dogs-double-all': [
        'hypernym-bananas', 'hypernym-bazookas', 'hypernym-cabinets', 'hypernym-cars',
        'hypernym-chairs', 'hypernym-crows', 'hypernym-diapers', 'hypernym-dogs',
        'hypernym-dolls', 'hypernym-ducklings', 'hypernym-elephants', 'hypernym-guns',
        'hypernym-hammers', 'hypernym-helmets', 'hypernym-jackets', 'hypernym-kayaks',
        'hypernym-kites', 'hypernym-mirrors',
    ],
    'hypernym-concat-bananas-to-dogs-double': [
        'hypernym-bananas', 'hypernym-bazookas', 'hypernym-cabinets', 'hypernym-cars',
        'hypernym-chairs', 'hypernym-crows', 'hypernym-diapers', 'hypernym-dogs',
        'hypernym-dolls', 'hypernym-ducklings', 'hypernym-elephants', 'hypernym-guns',
        'hypernym-hammers', 'hypernym-helmets', 'hypernym-jackets', 'hypernym-kayaks',
        'hypernym-kites', 'hypernym-mirrors',
    ],
}


def build_eval_commands(hf_org: str, repo_name: str, parsed: dict) -> str:
    """
    Build eval bash command(s) for the model card.

    Script/flag selection by typicality correction type:
      tc=self    -> eval_by_claude.py --self-typicality
      tc=online  -> eval.py --typicality-correction
      tc=neg     -> eval_by_claude.py --neg-typicality
      tc=None    -> eval_by_claude.py --self-typicality (always pass so CSV includes all variants)

    Concat tasks are expanded to their individual eval subtasks.
    """
    tc = parsed.get('tc')
    task_segment = parsed['task_segment']
    model_id = f"{hf_org}/{repo_name}"

    if tc == 'online':
        script = 'scripts/eval.py'
        tc_flag = '--typicality-correction'
    elif tc == 'neg':
        script = 'scripts/eval_by_claude.py'
        tc_flag = '--neg-typicality'
    else:
        # tc='self' or tc=None: always use --self-typicality so CSV includes all variants
        script = 'scripts/eval_by_claude.py'
        tc_flag = '--self-typicality'

    base_flags = '--split_type random --gen-shots zero --disc-shots few --validator-log-odds --save-scores-csv'

    tasks = CONCAT_TASK_SUBTASKS.get(task_segment, [task_segment])

    lines = []
    for task in tasks:
        parts = [f'python {script}', f'--model {model_id}', f'--task {task}', base_flags]
        if tc_flag:
            parts.append(tc_flag)
        lines.append(' \\\n    '.join(parts))

    return '\n\n'.join(lines)


MODEL_CARD_TEMPLATE = """\
---
library_name: transformers
base_model: {base_model}
tags:
  - rankalign
  - fine-tuned
---

# {repo_name}

Fine-tuned checkpoint from the [rankalign](https://github.com/juand-r/rankalign) project.

## Training Details

| Field | Value |
|-------|-------|
| Base model | `{base_model}` |
| Version | {version} |
| Task | `{task_segment}` |
| Epoch | {epoch} |
| Delta | {delta} |
| Typicality correction | {tc} |
| Length normalization | {lenorm} |
| Preference loss weight | {pref} |
| NLL validator weight | {nll_v} |
| NLL generator weight | {nll_g} |
| Validator log-odds | {vallogodds} |
| Force same-x | {force_same_x} |
| Semi-supervised ratio | {semi} |
| Labeled-only ratio | {labelonly} |

## Reproducibility

**Original checkpoint name:** `{original_name}`

To evaluate:
```bash
{eval_commands}
```
"""


def update_checkpoint_map(models_dir: Path, repo_name: str, original_name: str):
    """Append or update entry in models/hf_checkpoint_map.json."""
    map_file = models_dir / 'hf_checkpoint_map.json'
    mapping = {}
    if map_file.exists():
        with open(map_file) as f:
            mapping = json.load(f)
    mapping[repo_name] = original_name
    with open(map_file, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    log.info(f"Updated checkpoint map: {map_file}")


def update_hf_repos_md(notes_dir: Path, repo_name: str, hf_org: str, parsed: dict):
    """Append a new entry to HUGGINGFACE_REPOS.md in the experiment notes dir."""
    hf_repos_file = notes_dir / 'HUGGINGFACE_REPOS.md'
    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    url = f"https://huggingface.co/{hf_org}/{repo_name}"
    entry = (
        f"\n## {repo_name} ({date_str})\n"
        f"- **Type:** Model checkpoint\n"
        f"- **Epoch:** {parsed['epoch']}\n"
        f"- **Task:** {parsed['task_segment']}\n"
        f"- **Base model:** {parsed['model_name']}\n"
        f"- [Model checkpoint — epoch {parsed['epoch']}, {parsed['task_segment']} ({date_str})]({url})\n"
    )
    with open(hf_repos_file, 'a') as f:
        f.write(entry)
    log.info(f"Updated {hf_repos_file}")


def main():
    parser = argparse.ArgumentParser(description="Upload rankalign checkpoint to HuggingFace Hub")
    parser.add_argument('--local-path', required=True,
                        help='Local checkpoint directory to upload')
    parser.add_argument('--hf-org', required=True,
                        help='HuggingFace org to upload to (e.g. juand-r)')
    parser.add_argument('--experiment-notes-dir', default='',
                        help='Path to experiment notes dir for updating HUGGINGFACE_REPOS.md (optional)')
    args = parser.parse_args()

    local_path = Path(args.local_path).resolve()
    if not local_path.exists():
        log.error(f"Checkpoint path does not exist: {local_path}")
        sys.exit(1)

    # Parse checkpoint name and build HF repo name
    try:
        parsed = parse_checkpoint_name(str(local_path))
    except ValueError as e:
        log.error(f"Failed to parse checkpoint name: {e}")
        sys.exit(1)

    repo_name = to_hf_repo_name(parsed)
    repo_id = f"{args.hf_org}/{repo_name}"
    log.info(f"Uploading {local_path.name} → {repo_id}")

    # Create the repo if it doesn't exist
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type='model')
        log.info(f"Repo {repo_id} already exists, updating.")
    except RepositoryNotFoundError:
        log.info(f"Creating new model repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type='model', private=False, exist_ok=True)

    # Write model card (README.md) into the checkpoint dir before uploading
    model_card = MODEL_CARD_TEMPLATE.format(
        repo_name=repo_name,
        hf_org=args.hf_org,
        base_model=parsed['model_name'],
        version=parsed['version'],
        task_segment=parsed['task_segment'],
        epoch=parsed['epoch'],
        delta=f"{parsed['delta']:g}",
        tc=parsed['tc'] or 'none',
        lenorm=parsed['lenorm'],
        pref=f"{parsed['pref']:g}",
        nll_v=f"{parsed['nll_v']:g}",
        nll_g=f"{parsed['nll_g']:g}",
        vallogodds=parsed['vallogodds'],
        force_same_x=parsed['force_same_x'],
        semi=parsed['semi'],
        labelonly=parsed['labelonly'],
        original_name=parsed['original_name'],
        eval_commands=build_eval_commands(args.hf_org, repo_name, parsed),
    )
    (local_path / 'README.md').write_text(model_card)

    # Upload
    log.info(f"Uploading folder {local_path} ...")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(local_path),
        repo_type='model',
        commit_message=f"Upload checkpoint: epoch {parsed['epoch']} ({parsed['task_segment']})",
        ignore_patterns=['*.py', '*.sh', '__pycache__'],
    )
    log.info(f"Upload complete: https://huggingface.co/{repo_id}")

    # Update checkpoint map (models/hf_checkpoint_map.json)
    update_checkpoint_map(local_path.parent, repo_name, parsed['original_name'])

    # Optionally update HUGGINGFACE_REPOS.md
    if args.experiment_notes_dir:
        notes_dir = Path(args.experiment_notes_dir)
        if notes_dir.exists():
            update_hf_repos_md(notes_dir, repo_name, args.hf_org, parsed)
        else:
            log.warning(f"Notes dir not found, skipping HUGGINGFACE_REPOS.md: {notes_dir}")


if __name__ == '__main__':
    main()
