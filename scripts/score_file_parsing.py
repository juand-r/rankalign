"""
Shared filename-parsing utilities for scores CSV files.

Extracted from dashboard_viz_refactor.py so that other scripts (e.g.
summarize_scores.py) can reuse the same parsing logic without pulling in
Dash/Plotly dependencies.

Both dashboard_viz_refactor.py and summarize_scores.py import from here.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# FILEINFO DATACLASS
# =============================================================================

@dataclass
class FileInfo:
    """Parsed representation of a scores CSV filename."""
    path: str
    filename: str
    task: str               # e.g. "hypernym-bananas"
    dataset: str            # e.g. "bananas"
    split: str              # "train" or "test"
    is_base: bool           # no delta in filename
    direction: Optional[str]  # "d2g" or None (base models)
    is_union: bool          # has force-same-x + union training task
    # Training weights (None for base models or if absent):
    pref_weight: Optional[float]
    nllv_weight: Optional[float]
    nllg_weight: Optional[float]
    # Flags:
    has_tco: bool           # _tc-online_ in filename
    has_norm: bool          # _lenorm_ in filename
    has_vallogodds: bool    # _vallogodds in filename
    # Derived:
    training_mode: Optional[str]  # "SFT", "Pref", or "Comb" (None for base)
    category: str           # "Base", "S", or "U"
    row_label: str          # full label e.g. "S-Comb-tco-norm"
    timestamp: str          # extracted from filename for dedup


# =============================================================================
# LOW-LEVEL EXTRACTION HELPERS
# =============================================================================

def _extract_float(pattern, text):
    """Extract a float value from a regex pattern with a named group 'weight'."""
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group('weight'))
        except (ValueError, IndexError):
            return None
    return None


def _extract_timestamp(filename):
    """Extract timestamp from end of filename for dedup ordering."""
    match = re.search(r'(\d{8}_\d{6})\.csv$', filename)
    return match.group(1) if match else '00000000_000000'


def _extract_task(stem, task_pattern, union_config):
    """Extract task and dataset from filename stem.

    Uses the *last* regex match of task_pattern as the eval task (finetuned
    filenames repeat the task: once for training, once for eval).
    """
    # Check for union model first
    if union_config:
        training_task = union_config.get('training_task', '')
        eval_pattern = union_config.get('eval_task_pattern', '')
        if training_task and training_task in stem and 'force-same-x' in stem:
            if eval_pattern:
                eval_match = re.search(eval_pattern, stem)
                if eval_match and eval_match.groups():
                    full_match = re.search(r'(hypernym-[a-zA-Z]+)', eval_match.group(0))
                    if full_match:
                        task = full_match.group(1)
                        dataset = task.split('-')[-1] if '-' in task else task
                        return task, dataset

    # Regular task extraction
    all_matches = list(re.finditer(task_pattern, stem))
    if not all_matches:
        return None, None

    # Use last match as eval task
    match = all_matches[-1]
    if match.groups():
        base_task = task_pattern.split('(')[0].rstrip('-').rstrip('_')
        if not base_task:
            base_task = 'task'
        dataset = match.group(1)
        task = f"{base_task}-{dataset}" if base_task else dataset
    else:
        task = match.group(0)
        dataset = task
    return task, dataset


def _extract_split(stem, split_patterns):
    """Extract split (train/test) from filename stem."""
    for split_name, pattern in split_patterns.items():
        if pattern in stem:
            return split_name
    return 'unknown'


# =============================================================================
# ROW LABEL BUILDER
# =============================================================================

def build_row_label(category, training_mode, has_tco, has_norm, has_vallogodds):
    """Build a row label from parsed fields.

    Format: {category}-{mode}[-tco][-norm][-v]
    Examples: "Base", "S-Comb", "S-Comb-tco-norm", "U-SFT-v"
    """
    if category == 'Base':
        return 'Base'

    parts = [f"{category}-{training_mode}"]
    if has_tco:
        parts.append('tco')
    if has_norm:
        parts.append('norm')
    if has_vallogodds:
        parts.append('v')
    return '-'.join(parts)


def row_sort_key(row_label):
    """Sort key for row labels. Base always last (bottom of heatmap)."""
    if row_label == 'Base':
        return (3, '', '', '')

    parts = row_label.split('-', 2)
    category = parts[0] if parts else ''
    cat_order = 0 if category == 'S' else 1
    mode = parts[1] if len(parts) > 1 else ''
    mode_order = {'Comb': 0, 'SFT': 1, 'Pref': 2}.get(mode, 3)
    flags = '-'.join(parts[2:]) if len(parts) > 2 else ''
    return (cat_order, mode_order, flags, row_label)


# =============================================================================
# FULL FILENAME PARSER (dashboard-oriented)
# =============================================================================

def parse_filename(csv_file, config):
    """Parse a scores CSV filename into a FileInfo.

    5-step pipeline:
      1. Base vs Finetuned (delta check)
      2. Direction filter (must be d2g for finetuned)
      3. U(nion) vs S(ingle)
      4. Training mode (SFT / Pref / Comb) - mutually exclusive
      5. Flags (tco, norm, v)

    Returns FileInfo or None (for files that should be skipped).
    Raises ValueError for unexpected finetuned filename formats.
    """
    name = Path(csv_file).name
    stem = Path(csv_file).stem
    task_pattern = config.get('task_pattern', r'hypernym-([a-zA-Z]+)')
    base_pattern = config.get('base_pattern')
    finetuned_pattern = config.get('finetuned_pattern')
    union_config = config.get('union_models', {})

    # --- Extract task and split ---
    task, dataset = _extract_task(stem, task_pattern, union_config)
    if task is None:
        return None

    split = _extract_split(stem, config.get('split_patterns', {}))
    if split == 'unknown':
        return None

    timestamp = _extract_timestamp(name)

    # --- Step 1: Base vs Finetuned ---
    is_finetuned = bool(finetuned_pattern and re.match(finetuned_pattern, stem))
    is_base = bool(base_pattern and re.match(base_pattern, stem) and not is_finetuned)

    if is_base:
        return FileInfo(
            path=str(csv_file), filename=name,
            task=task, dataset=dataset, split=split,
            is_base=True, direction=None, is_union=False,
            pref_weight=None, nllv_weight=None, nllg_weight=None,
            has_tco=False, has_norm=False, has_vallogodds=False,
            training_mode=None, category='Base', row_label='Base',
            timestamp=timestamp,
        )

    if not is_finetuned:
        return None

    # --- Step 2: Direction filter (d2g only) ---
    if '_d2g_' not in stem:
        return None

    # --- Step 3: U(nion) vs S(ingle) ---
    is_union = False
    if union_config:
        training_task = union_config.get('training_task', '')
        eval_pattern = union_config.get('eval_task_pattern', '')
        if training_task and training_task in stem and 'force-same-x' in stem:
            is_union = True
            if eval_pattern:
                eval_match = re.search(eval_pattern, stem)
                if eval_match and eval_match.groups():
                    full_match = re.search(r'(hypernym-[a-zA-Z]+)', eval_match.group(0))
                    if full_match:
                        task = full_match.group(1)
                        dataset = task.split('-')[-1] if '-' in task else task

    # For non-union finetuned models, verify training and eval tasks match
    if not is_union:
        all_matches = list(re.finditer(task_pattern, stem))
        if len(all_matches) >= 2:
            training_match = all_matches[0]
            eval_match = all_matches[-1]
            if training_match.groups() and eval_match.groups():
                if training_match.group(1) != eval_match.group(1):
                    return None

    category = 'U' if is_union else 'S'

    # --- Step 4: Training mode ---
    pref_weight = _extract_float(r'(?:^|[_-])pref(?P<weight>\d+(?:\.\d+)?)', stem)
    nllv_weight = _extract_float(r'nllv(?P<weight>\d+(?:\.\d+)?)', stem)
    nllg_weight = _extract_float(r'nllg(?P<weight>\d+(?:\.\d+)?)', stem)

    eff_pref = pref_weight if pref_weight is not None else 1.0
    eff_nllv = nllv_weight if nllv_weight is not None else 0.0
    eff_nllg = nllg_weight if nllg_weight is not None else 0.0

    is_sft = (eff_pref == 0.0 and eff_nllv == 1.0 and eff_nllg == 1.0)
    is_pref_only = (eff_nllv == 0.0 and eff_nllg == 0.0)
    is_pref_nll = (eff_nllv == 1.0 and eff_nllg == 1.0 and eff_pref == 1.0)

    modes_matched = sum([is_sft, is_pref_only, is_pref_nll])
    if modes_matched != 1:
        raise ValueError(
            f"Training mode ambiguous or unrecognized for finetuned file: {name}\n"
            f"  pref={pref_weight}, nllv={nllv_weight}, nllg={nllg_weight}\n"
            f"  Matched: SFT={is_sft}, Pref={is_pref_only}, Comb={is_pref_nll}"
        )

    if is_sft:
        training_mode = 'SFT'
    elif is_pref_only:
        training_mode = 'Pref'
    else:
        training_mode = 'Comb'

    # --- Step 5: Flags ---
    has_tco = '_tc-online_' in stem
    has_norm = '_lenorm_' in stem
    has_vallogodds = '_vallogodds' in stem

    row_label = build_row_label(category, training_mode, has_tco, has_norm, has_vallogodds)

    return FileInfo(
        path=str(csv_file), filename=name,
        task=task, dataset=dataset, split=split,
        is_base=False, direction='d2g', is_union=is_union,
        pref_weight=pref_weight, nllv_weight=nllv_weight, nllg_weight=nllg_weight,
        has_tco=has_tco, has_norm=has_norm, has_vallogodds=has_vallogodds,
        training_mode=training_mode, category=category, row_label=row_label,
        timestamp=timestamp,
    )


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def discover_and_resolve_files(config):
    """Discover all scores files and parse them into FileInfo objects.

    Returns:
        list of FileInfo: All successfully parsed files.
    """
    outputs_dir = Path(config['outputs_dir'])
    file_pattern = config.get('file_pattern', 'scores_*.csv')

    all_files = []
    skipped = 0
    for csv_file in sorted(outputs_dir.glob(file_pattern)):
        try:
            info = parse_filename(csv_file, config)
            if info is not None:
                all_files.append(info)
            else:
                skipped += 1
        except ValueError as e:
            raise ValueError(str(e))

    print(f"Discovered {len(all_files)} valid files, skipped {skipped}")
    return all_files


def resolve_files_for_task(file_infos, task, split):
    """Resolve files for a specific task/split to a 1:1 row_label -> FileInfo mapping.

    If multiple files map to the same row_label, picks the newest by timestamp.

    Returns:
        dict: {row_label: FileInfo}
        list: warnings
    """
    matching = [f for f in file_infos if f.task == task and f.split == split]

    by_label = {}
    for f in matching:
        if f.row_label not in by_label:
            by_label[f.row_label] = []
        by_label[f.row_label].append(f)

    result = {}
    warnings = []
    for label, files in by_label.items():
        if len(files) == 1:
            result[label] = files[0]
        else:
            files_sorted = sorted(files, key=lambda f: f.timestamp, reverse=True)
            result[label] = files_sorted[0]
            filenames = [f.filename for f in files_sorted]
            warnings.append(
                f"  [DEDUP] {task}/{split}/{label}: {len(files)} files found, "
                f"using newest: {filenames[0]} (dropped: {filenames[1:]})"
            )

    return result, warnings
