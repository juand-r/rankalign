#!/usr/bin/env python3
"""
Generate LaTeX tables for semi-supervised experiment results.

Tables:
  - PlausibleQA: correlation and gen ROC (averaged over 100 test prompts)
  - AmbigQA: correlation and gen ROC (averaged over 17 test prompts)
  - Hypernym in-domain: correlation and gen ROC (averaged over 8 in-domain nouns)
  - Hypernym out-of-domain: correlation and gen ROC (averaged over 10 OOD nouns)

Rows: model variants (training recipe x training-time flags)
Columns: eval mode (raw, tc, lenorm, tc+lenorm)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from score_file_parsing import _extract_timestamp

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / 'outputs'

# Strict integrity checks:
# - latest batch only (per variant, then family-wide consistency for semisupervised variants)
# - full task coverage required for every row before table generation
STRICT_LATEST_BATCH_ONLY = False
STRICT_FULL_COVERAGE = True

EVAL_COLUMNS = {
    'raw': 'gen_score',
    'tc': 'gen_score_typcorr',
    'lenorm': 'gen_score_lenorm',
    'tc+lenorm': 'gen_score_typcorr_lenorm',
}

LABEL_COLUMNS = [
    ('gpt4_ground_truth', {'yes': 1, 'no': 0}),
    ('correct', {'yes': 1, 'no': 0}),
]

HYPERNYM_IN_DOMAIN = {
    'bananas', 'bazookas', 'cabinets', 'cars', 'chairs', 'crows', 'diapers', 'dogs'
}

PLAUSIBLEQA_TASKS = [
    'plausibleqa-nq_1114', 'plausibleqa-nq_1324', 'plausibleqa-nq_1328',
    'plausibleqa-nq_1369', 'plausibleqa-nq_1394', 'plausibleqa-nq_1438',
    'plausibleqa-nq_1663', 'plausibleqa-nq_2031', 'plausibleqa-nq_207',
    'plausibleqa-nq_2174', 'plausibleqa-nq_2281', 'plausibleqa-nq_2421',
    'plausibleqa-nq_2436', 'plausibleqa-nq_2535', 'plausibleqa-nq_2622',
    'plausibleqa-nq_2637', 'plausibleqa-nq_2759', 'plausibleqa-nq_2824',
    'plausibleqa-nq_2856', 'plausibleqa-nq_2867', 'plausibleqa-nq_2876',
    'plausibleqa-nq_3004', 'plausibleqa-nq_3015', 'plausibleqa-nq_3068',
    'plausibleqa-nq_3099', 'plausibleqa-nq_3127', 'plausibleqa-nq_3137',
    'plausibleqa-nq_316', 'plausibleqa-nq_3276', 'plausibleqa-nq_54',
    'plausibleqa-nq_562', 'plausibleqa-nq_709', 'plausibleqa-nq_958',
    'plausibleqa-trivia_1655', 'plausibleqa-trivia_2984', 'plausibleqa-trivia_3035',
    'plausibleqa-trivia_3043', 'plausibleqa-trivia_3180', 'plausibleqa-trivia_3245',
    'plausibleqa-trivia_3433', 'plausibleqa-trivia_3492', 'plausibleqa-trivia_3599',
    'plausibleqa-trivia_4009', 'plausibleqa-trivia_4234', 'plausibleqa-trivia_4489',
    'plausibleqa-trivia_4697', 'plausibleqa-trivia_5003', 'plausibleqa-trivia_560',
    'plausibleqa-trivia_5675', 'plausibleqa-trivia_6317', 'plausibleqa-trivia_6777',
    'plausibleqa-trivia_7272', 'plausibleqa-trivia_7579', 'plausibleqa-trivia_9589',
    'plausibleqa-webq_1000', 'plausibleqa-webq_1046', 'plausibleqa-webq_1086',
    'plausibleqa-webq_1097', 'plausibleqa-webq_1163', 'plausibleqa-webq_1187',
    'plausibleqa-webq_1278', 'plausibleqa-webq_1307', 'plausibleqa-webq_1310',
    'plausibleqa-webq_1338', 'plausibleqa-webq_134', 'plausibleqa-webq_1383',
    'plausibleqa-webq_141', 'plausibleqa-webq_1421', 'plausibleqa-webq_1442',
    'plausibleqa-webq_1476', 'plausibleqa-webq_1498', 'plausibleqa-webq_15',
    'plausibleqa-webq_1584', 'plausibleqa-webq_1613', 'plausibleqa-webq_1668',
    'plausibleqa-webq_1714', 'plausibleqa-webq_1723', 'plausibleqa-webq_1836',
    'plausibleqa-webq_1972', 'plausibleqa-webq_212', 'plausibleqa-webq_299',
    'plausibleqa-webq_342', 'plausibleqa-webq_373', 'plausibleqa-webq_428',
    'plausibleqa-webq_435', 'plausibleqa-webq_520', 'plausibleqa-webq_611',
    'plausibleqa-webq_650', 'plausibleqa-webq_669', 'plausibleqa-webq_672',
    'plausibleqa-webq_713', 'plausibleqa-webq_744', 'plausibleqa-webq_749',
    'plausibleqa-webq_760', 'plausibleqa-webq_77', 'plausibleqa-webq_803',
    'plausibleqa-webq_84', 'plausibleqa-webq_88', 'plausibleqa-webq_882',
    'plausibleqa-webq_898',
]

AMBIGQA_TASKS = [
    'ambigqa-american', 'ambigqa-danube', 'ambigqa-executed', 'ambigqa-gives',
    'ambigqa-harry', 'ambigqa-involved', 'ambigqa-jack', 'ambigqa-plays',
    'ambigqa-received', 'ambigqa-sang', 'ambigqa-soccer', 'ambigqa-used',
    'ambigqa-voice', 'ambigqa-winter', 'ambigqa-won', 'ambigqa-world', 'ambigqa-year',
]

HYPERNYM_TASKS = [
    'hypernym-bananas', 'hypernym-bazookas', 'hypernym-cabinets', 'hypernym-cars',
    'hypernym-chairs', 'hypernym-crows', 'hypernym-diapers', 'hypernym-dogs',
    'hypernym-dolls', 'hypernym-ducklings', 'hypernym-elephants', 'hypernym-guns',
    'hypernym-hammers', 'hypernym-helmets', 'hypernym-jackets', 'hypernym-kayaks',
    'hypernym-kites', 'hypernym-mirrors',
]

IFEVAL_TASKS = [
    'ifeval-prompt_1', 'ifeval-prompt_2', 'ifeval-prompt_3', 'ifeval-prompt_4',
    'ifeval-prompt_5', 'ifeval-prompt_6', 'ifeval-prompt_7', 'ifeval-prompt_8',
    'ifeval-prompt_9', 'ifeval-prompt_10', 'ifeval-prompt_11', 'ifeval-prompt_12',
    'ifeval-prompt_13', 'ifeval-prompt_15', 'ifeval-prompt_16', 'ifeval-prompt_17',
    'ifeval-prompt_18', 'ifeval-prompt_19', 'ifeval-prompt_20', 'ifeval-prompt_21',
    'ifeval-prompt_22', 'ifeval-prompt_23', 'ifeval-prompt_24', 'ifeval-prompt_25',
    'ifeval-prompt_26', 'ifeval-prompt_27', 'ifeval-prompt_28', 'ifeval-prompt_29',
    'ifeval-prompt_30', 'ifeval-prompt_32', 'ifeval-prompt_33', 'ifeval-prompt_34',
    'ifeval-prompt_35', 'ifeval-prompt_36', 'ifeval-prompt_37', 'ifeval-prompt_38',
    'ifeval-prompt_39', 'ifeval-prompt_40', 'ifeval-prompt_41', 'ifeval-prompt_42',
    'ifeval-prompt_43', 'ifeval-prompt_44', 'ifeval-prompt_45', 'ifeval-prompt_46',
    'ifeval-prompt_47', 'ifeval-prompt_48', 'ifeval-prompt_49', 'ifeval-prompt_50',
    'ifeval-prompt_51', 'ifeval-prompt_52', 'ifeval-prompt_53', 'ifeval-prompt_54',
    'ifeval-prompt_56', 'ifeval-prompt_57', 'ifeval-prompt_58', 'ifeval-prompt_59',
    'ifeval-prompt_61', 'ifeval-prompt_63', 'ifeval-prompt_64', 'ifeval-prompt_65',
    'ifeval-prompt_66', 'ifeval-prompt_67', 'ifeval-prompt_68', 'ifeval-prompt_70',
    'ifeval-prompt_72', 'ifeval-prompt_73', 'ifeval-prompt_74', 'ifeval-prompt_75',
    'ifeval-prompt_76', 'ifeval-prompt_77', 'ifeval-prompt_78', 'ifeval-prompt_79',
    'ifeval-prompt_80', 'ifeval-prompt_82', 'ifeval-prompt_83', 'ifeval-prompt_84',
    'ifeval-prompt_85', 'ifeval-prompt_87', 'ifeval-prompt_88', 'ifeval-prompt_89',
    'ifeval-prompt_90', 'ifeval-prompt_91', 'ifeval-prompt_92', 'ifeval-prompt_93',
    'ifeval-prompt_94', 'ifeval-prompt_95', 'ifeval-prompt_96', 'ifeval-prompt_97',
    'ifeval-prompt_98', 'ifeval-prompt_99', 'ifeval-prompt_100', 'ifeval-prompt_102',
    'ifeval-prompt_103', 'ifeval-prompt_104', 'ifeval-prompt_105', 'ifeval-prompt_106',
    'ifeval-prompt_107', 'ifeval-prompt_108', 'ifeval-prompt_109',
]

# ── Model variant definitions ──
# Each entry: (glob_pattern_fragment, human_label, sort_order)
# The fragment matches the part after alpha1.0_ in the self- prefixed filenames.
# For base model, separate handling.

SEMI_VARIANTS_AMBIGQA = [
    # PLAIN
    ('full-completion_force-same-x_labelonly0.1', 'Pref, label-only', 'plain', 0),
    ('full-completion_force-same-x_vallogodds_labelonly0.1', 'Pref+v, label-only', 'plain', 1),
    ('full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_labelonly0.1', 'Comb+v, label-only', 'plain', 2),
    ('full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1', 'Comb+v, semi', 'plain', 3),
    ('full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x_labelonly0.1', 'SFT, label-only', 'plain', 4),
    ('full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x_semi0.1', 'SFT, semi', 'plain', 5),
    # TC-SELF
    ('tc-self_full-completion_force-same-x_labelonly0.1', 'Pref, label-only', 'tc-self', 10),
    ('tc-self_full-completion_force-same-x_vallogodds_labelonly0.1', 'Pref+v, label-only', 'tc-self', 11),
    ('tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_labelonly0.1', 'Comb+v, label-only', 'tc-self', 12),
    ('tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1', 'Comb+v, semi', 'tc-self', 13),
    ('tc-self_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x_semi0.1', 'SFT, semi', 'tc-self', 14),
    # TC-SELF + LENORM
    ('tc-self_lenorm_full-completion_force-same-x_labelonly0.1', 'Pref, label-only', 'tc-self+len', 20),
    ('tc-self_lenorm_full-completion_force-same-x_vallogodds_labelonly0.1', 'Pref+v, label-only', 'tc-self+len', 21),
    ('tc-self_lenorm_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_labelonly0.1', 'Comb+v, label-only', 'tc-self+len', 22),
    ('tc-self_lenorm_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1', 'Comb+v, semi', 'tc-self+len', 23),
    ('tc-self_lenorm_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x_semi0.1', 'SFT, semi', 'tc-self+len', 24),
]


def _extract_batch_date(path):
    ts = _extract_timestamp(path.name)
    if not ts or ts == '00000000_000000':
        return ''
    return ts.split('_')[0]


def _summarize_tasks(tasks, max_items=10):
    tasks = sorted(tasks)
    if len(tasks) <= max_items:
        return ', '.join(tasks)
    shown = ', '.join(tasks[:max_items])
    return f"{shown}, ... (+{len(tasks) - max_items} more)"


def _score_file_pattern(task_family, training_task_fragment, variant_fragment, task_name):
    test_suffix = '_test_v2_log-odds' if task_family == 'hypernym' else '_test_log-odds'
    if variant_fragment == '__BASE__':
        return f"scores_self-v6-google_gemma-2-2b_{task_name}{test_suffix}_*tc_*.csv"
    return (
        f"scores_self-v6-google_gemma-2-2b-delta0.15-epoch*_"
        f"{training_task_fragment}_d2g_random_alpha1.0_"
        f"{variant_fragment}_{task_name}{test_suffix}_*tc_*.csv"
    )


def _resolve_variant_task_files(task_family, training_task_fragment, variant_fragment, task_list):
    """
    Resolve a 1:1 mapping task -> file for one variant.

    Strict behavior:
    - If STRICT_LATEST_BATCH_ONLY: only files from this variant's latest batch date are allowed.
    - If STRICT_FULL_COVERAGE: every task must have a selected file.
    """
    candidates_by_task = {}
    for task_name in task_list:
        pattern = _score_file_pattern(task_family, training_task_fragment, variant_fragment, task_name)
        candidates_by_task[task_name] = sorted(OUTPUTS_DIR.glob(pattern))

    missing_any = [t for t, files in candidates_by_task.items() if not files]
    if missing_any:
        raise RuntimeError(
            f"[{task_family}] Variant '{variant_fragment}' has no files for "
            f"{len(missing_any)}/{len(task_list)} tasks: {_summarize_tasks(missing_any)}"
        )

    selected = {}
    selected_batch = ''

    if STRICT_LATEST_BATCH_ONLY:
        all_batch_dates = sorted({
            _extract_batch_date(path)
            for files in candidates_by_task.values()
            for path in files
            if _extract_batch_date(path)
        })
        if not all_batch_dates:
            raise RuntimeError(
                f"[{task_family}] Variant '{variant_fragment}' has files but no parseable timestamps."
            )

        selected_batch = all_batch_dates[-1]
        missing_in_latest_batch = []
        for task_name, files in candidates_by_task.items():
            batch_files = [p for p in files if _extract_batch_date(p) == selected_batch]
            if not batch_files:
                missing_in_latest_batch.append(task_name)
                continue
            selected[task_name] = batch_files[-1]  # newest timestamp inside latest batch date

        if missing_in_latest_batch:
            raise RuntimeError(
                f"[{task_family}] Variant '{variant_fragment}' latest batch {selected_batch} is incomplete: "
                f"{len(task_list) - len(missing_in_latest_batch)}/{len(task_list)} tasks present; "
                f"missing: {_summarize_tasks(missing_in_latest_batch)}"
            )
    else:
        for task_name, files in candidates_by_task.items():
            selected[task_name] = files[-1]

    if STRICT_FULL_COVERAGE and len(selected) != len(task_list):
        missing = sorted(set(task_list) - set(selected.keys()))
        raise RuntimeError(
            f"[{task_family}] Variant '{variant_fragment}' incomplete coverage: "
            f"{len(selected)}/{len(task_list)} tasks; missing: {_summarize_tasks(missing)}"
        )

    return selected, selected_batch


def _resolve_family_variant_files(task_family, training_task_fragment, variants, full_task_list, skip_base=False):
    """
    Resolve files for all variants in one family, enforcing strict consistency.
    """
    variant_fragments = [v[0] for v in variants] if skip_base else ['__BASE__'] + [v[0] for v in variants]
    resolved = {}
    errors = []

    for variant_fragment in variant_fragments:
        try:
            task_files, batch_date = _resolve_variant_task_files(
                task_family, training_task_fragment, variant_fragment, full_task_list
            )
            resolved[variant_fragment] = {
                'task_files': task_files,
                'batch_date': batch_date,
            }
        except RuntimeError as exc:
            errors.append(str(exc))

    if errors:
        joined = '\n'.join(f"  - {msg}" for msg in errors)
        raise RuntimeError(
            f"[{task_family}] Coverage validation failed for {len(errors)} variant(s):\n{joined}"
        )

    # Enforce a single latest batch date across all semisupervised variants (base excluded).
    if STRICT_LATEST_BATCH_ONLY:
        semi_batches = {
            frag: info['batch_date']
            for frag, info in resolved.items()
            if frag != '__BASE__'
        }
        latest_family_batch = max(semi_batches.values())
        stale = {frag: date for frag, date in semi_batches.items() if date != latest_family_batch}
        if stale:
            stale_preview = ', '.join(f"{k}:{v}" for k, v in sorted(stale.items())[:8])
            if len(stale) > 8:
                stale_preview += f", ... (+{len(stale) - 8} more)"
            raise RuntimeError(
                f"[{task_family}] Mixed semisupervised batch dates detected. "
                f"Latest is {latest_family_batch}, but stale variants exist: {stale_preview}"
            )

    return resolved


def load_labels(df):
    for col_name, label_map in LABEL_COLUMNS:
        if col_name not in df.columns:
            continue
        gt = df[col_name]
        if gt.dtype in ('int64', 'float64', 'int', 'float'):
            return gt.astype(int).values
        mapped = gt.str.strip().str.lower().map(label_map)
        if mapped.isna().all():
            continue
        return mapped.fillna(0).astype(int).values
    return None


def compute_metrics(df):
    """Compute gen_roc and correlation for each eval variant."""
    labels = load_labels(df)
    if labels is None:
        return None

    val_scores = df['val_score'].values
    results = {}

    for eval_name, gen_col in EVAL_COLUMNS.items():
        if gen_col not in df.columns:
            results[eval_name] = {'gen_roc': np.nan, 'corr': np.nan}
            continue

        gen_scores = df[gen_col].values
        valid = ~(np.isnan(gen_scores) | np.isnan(val_scores))
        if valid.sum() < 2:
            results[eval_name] = {'gen_roc': np.nan, 'corr': np.nan}
            continue

        gen_v = gen_scores[valid]
        val_v = val_scores[valid]
        lab_v = labels[valid]

        try:
            gen_roc = roc_auc_score(lab_v, gen_v)
        except Exception:
            gen_roc = np.nan

        try:
            corr, _ = pearsonr(gen_v, val_v)
        except Exception:
            corr = np.nan

        results[eval_name] = {'gen_roc': gen_roc, 'corr': corr}

    return results


def find_files_for_variant(task_family, training_task_fragment, variant_fragment, task_name, batch_date=None):
    """
    Convenience lookup used by debug/audit code.
    Returns newest matching file, optionally restricted to a specific batch date (YYYYMMDD).
    """
    pattern = _score_file_pattern(task_family, training_task_fragment, variant_fragment, task_name)
    matches = sorted(OUTPUTS_DIR.glob(pattern))
    if batch_date is not None:
        matches = [m for m in matches if _extract_batch_date(m) == batch_date]
    if not matches:
        return None
    return matches[-1]


def _aggregate_metrics_from_task_files(task_files, family_label, variant_label):
    """Compute averaged metrics from an explicit task->file mapping (strict, no silent skips)."""
    all_metrics = {ev: {'gen_roc': [], 'corr': []} for ev in EVAL_COLUMNS}

    for task_name, csv_path in task_files.items():
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(
                f"[{family_label}] Failed to read {csv_path.name} "
                f"for variant '{variant_label}', task '{task_name}': {exc}"
            ) from exc

        metrics = compute_metrics(df)
        if metrics is None:
            raise RuntimeError(
                f"[{family_label}] Could not compute metrics for {csv_path.name} "
                f"(variant '{variant_label}', task '{task_name}'): labels missing."
            )

        for ev in EVAL_COLUMNS:
            gen_roc = metrics[ev]['gen_roc']
            corr = metrics[ev]['corr']
            if np.isnan(gen_roc) or np.isnan(corr):
                raise RuntimeError(
                    f"[{family_label}] NaN metric for variant '{variant_label}', task '{task_name}', "
                    f"eval mode '{ev}' in file {csv_path.name}."
                )
            all_metrics[ev]['gen_roc'].append(gen_roc)
            all_metrics[ev]['corr'].append(corr)

    result = {}
    for ev in EVAL_COLUMNS:
        result[ev] = {
            'gen_roc': float(np.mean(all_metrics[ev]['gen_roc'])),
            'corr': float(np.mean(all_metrics[ev]['corr'])),
            'n': len(all_metrics[ev]['gen_roc']),
        }
    return result


def build_table_data_from_resolved(task_family, variants, task_list, resolved_family_files):
    """Build table rows using pre-resolved strict task file mappings."""
    rows = []

    if '__BASE__' in resolved_family_files:
        base_task_files = {
            task: resolved_family_files['__BASE__']['task_files'][task]
            for task in task_list
        }
        base_metrics = _aggregate_metrics_from_task_files(base_task_files, task_family, '__BASE__')
        rows.append(('Base model', '', -1, base_metrics))

    for variant_frag, label, train_group, sort_key in variants:
        variant_task_files = {
            task: resolved_family_files[variant_frag]['task_files'][task]
            for task in task_list
        }
        metrics = _aggregate_metrics_from_task_files(variant_task_files, task_family, variant_frag)
        rows.append((label, train_group, sort_key, metrics))

    return rows


def format_val(v, bold_threshold=None):
    if np.isnan(v):
        return '---'
    return f'{v:.3f}'


def make_latex_table(rows, metric_key, caption, label):
    """Generate a LaTeX table string.
    
    rows: list of (label, train_group, sort_key, metrics_dict)
    metric_key: 'gen_roc' or 'corr'
    """
    eval_modes = list(EVAL_COLUMNS.keys())
    col_headers = ['raw', 'tc', 'lenorm', 'tc+lenorm']

    # Find best value per column (excluding base) for bolding
    best_per_col = {}
    for ev in eval_modes:
        vals = []
        for label_name, tg, sk, m in rows:
            if label_name == 'Base model':
                continue
            v = m[ev][metric_key]
            if not np.isnan(v):
                vals.append(v)
        best_per_col[ev] = max(vals) if vals else None

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{ll' + 'c' * len(eval_modes) + '}')
    lines.append(r'\toprule')
    lines.append(r'Train config & Loss / data & ' + ' & '.join(col_headers) + r' \\')
    lines.append(r'\midrule')

    current_group = None
    for label_name, train_group, sort_key, m in rows:
        if label_name == 'Base model':
            vals = []
            for ev in eval_modes:
                v = m[ev][metric_key]
                vals.append(format_val(v))
            lines.append(r'\multicolumn{2}{l}{\textit{Base model}} & ' + ' & '.join(vals) + r' \\')
            lines.append(r'\midrule')
            continue

        if train_group != current_group:
            if current_group is not None:
                lines.append(r'\midrule')
            current_group = train_group

        group_display = {
            'plain': 'Plain',
            'tc-self': 'TC-self',
            'tc-self+len': 'TC-self+len',
        }.get(train_group, train_group)

        vals = []
        for ev in eval_modes:
            v = m[ev][metric_key]
            s = format_val(v)
            if best_per_col[ev] is not None and not np.isnan(v) and abs(v - best_per_col[ev]) < 1e-6:
                s = r'\textbf{' + s + '}'
            vals.append(s)

        lines.append(f'{group_display} & {label_name} & ' + ' & '.join(vals) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def _semi_batch_date(resolved_family_files):
    semi_dates = {
        info['batch_date']
        for frag, info in resolved_family_files.items()
        if frag != '__BASE__'
    }
    if not semi_dates:
        return ''
    return sorted(semi_dates)[-1]


def generate_all_tables():
    """Generate all LaTeX tables and return the full document."""

    # ── PlausibleQA ──
    print("Resolving PlausibleQA files (strict mode)...")
    plausibleqa_resolved = _resolve_family_variant_files(
        'plausibleqa', 'plausibleqa-all', SEMI_VARIANTS_AMBIGQA, PLAUSIBLEQA_TASKS
    )
    print(f"  PlausibleQA semisupervised batch date: {_semi_batch_date(plausibleqa_resolved)}")
    plausibleqa_rows = build_table_data_from_resolved(
        'plausibleqa', SEMI_VARIANTS_AMBIGQA, PLAUSIBLEQA_TASKS, plausibleqa_resolved
    )

    # ── AmbigQA ──
    print("Resolving AmbigQA files (strict mode)...")
    ambigqa_resolved = _resolve_family_variant_files(
        'ambigqa', 'ambigqa-all', SEMI_VARIANTS_AMBIGQA, AMBIGQA_TASKS
    )
    print(f"  AmbigQA semisupervised batch date: {_semi_batch_date(ambigqa_resolved)}")
    ambigqa_rows = build_table_data_from_resolved(
        'ambigqa', SEMI_VARIANTS_AMBIGQA, AMBIGQA_TASKS, ambigqa_resolved
    )

    # ── Hypernym (in-domain) ──
    hypernym_id_tasks = [t for t in HYPERNYM_TASKS if t.split('-')[1] in HYPERNYM_IN_DOMAIN]
    hypernym_ood_tasks = [t for t in HYPERNYM_TASKS if t.split('-')[1] not in HYPERNYM_IN_DOMAIN]

    print(f"Hypernym in-domain tasks ({len(hypernym_id_tasks)}): {[t.split('-')[1] for t in hypernym_id_tasks]}")
    print(f"Hypernym OOD tasks ({len(hypernym_ood_tasks)}): {[t.split('-')[1] for t in hypernym_ood_tasks]}")

    print("Resolving Hypernym files (strict mode)...")
    hypernym_resolved = _resolve_family_variant_files(
        'hypernym', 'hypernym-concat-bananas-to-dogs-double-all',
        SEMI_VARIANTS_AMBIGQA, HYPERNYM_TASKS
    )
    print(f"  Hypernym semisupervised batch date: {_semi_batch_date(hypernym_resolved)}")

    print("Processing Hypernym in-domain...")
    hypernym_id_rows = build_table_data_from_resolved(
        'hypernym', SEMI_VARIANTS_AMBIGQA, hypernym_id_tasks, hypernym_resolved
    )

    print("Processing Hypernym out-of-domain...")
    hypernym_ood_rows = build_table_data_from_resolved(
        'hypernym', SEMI_VARIANTS_AMBIGQA, hypernym_ood_tasks, hypernym_resolved
    )

    # ── IFEval ──
    print(f"Resolving IFEval files (strict mode)... ({len(IFEVAL_TASKS)} tasks)")
    ifeval_resolved = _resolve_family_variant_files(
        'ifeval', 'ifeval-concat-all', SEMI_VARIANTS_AMBIGQA, IFEVAL_TASKS, skip_base=True
    )
    print(f"  IFEval semisupervised batch date: {_semi_batch_date(ifeval_resolved)}")
    ifeval_rows = build_table_data_from_resolved(
        'ifeval', SEMI_VARIANTS_AMBIGQA, IFEVAL_TASKS, ifeval_resolved
    )

    # ── Build LaTeX document ──
    tables = {}

    tables['pqa_corr'] = make_latex_table(
        plausibleqa_rows, 'corr',
        'PlausibleQA: Pearson correlation between generator and validator scores (avg.\\ over 100 test prompts).',
        'tab:plausibleqa-corr'
    )

    tables['pqa_roc'] = make_latex_table(
        plausibleqa_rows, 'gen_roc',
        'PlausibleQA: Generator ROC AUC (avg.\\ over 100 test prompts).',
        'tab:plausibleqa-genroc'
    )

    tables['ambigqa_corr'] = make_latex_table(
        ambigqa_rows, 'corr',
        'AmbigQA: Pearson correlation between generator and validator scores (avg.\\ over 17 test prompts).',
        'tab:ambigqa-corr'
    )

    tables['ambigqa_roc'] = make_latex_table(
        ambigqa_rows, 'gen_roc',
        'AmbigQA: Generator ROC AUC (avg.\\ over 17 test prompts).',
        'tab:ambigqa-genroc'
    )

    tables['hyp_id_corr'] = make_latex_table(
        hypernym_id_rows, 'corr',
        'Hypernym (in-domain): Pearson correlation (avg.\\ over 8 in-domain nouns).',
        'tab:hypernym-id-corr'
    )

    tables['hyp_id_roc'] = make_latex_table(
        hypernym_id_rows, 'gen_roc',
        'Hypernym (in-domain): Generator ROC AUC (avg.\\ over 8 in-domain nouns).',
        'tab:hypernym-id-genroc'
    )

    tables['hyp_ood_corr'] = make_latex_table(
        hypernym_ood_rows, 'corr',
        'Hypernym (out-of-domain): Pearson correlation (avg.\\ over 10 OOD nouns).',
        'tab:hypernym-ood-corr'
    )

    tables['hyp_ood_roc'] = make_latex_table(
        hypernym_ood_rows, 'gen_roc',
        'Hypernym (out-of-domain): Generator ROC AUC (avg.\\ over 10 OOD nouns).',
        'tab:hypernym-ood-genroc'
    )

    tables['ifeval_corr'] = make_latex_table(
        ifeval_rows, 'corr',
        f'IFEval: Pearson correlation between generator and validator scores (avg.\\ over {len(IFEVAL_TASKS)} test prompts).',
        'tab:ifeval-corr'
    )

    tables['ifeval_roc'] = make_latex_table(
        ifeval_rows, 'gen_roc',
        f'IFEval: Generator ROC AUC (avg.\\ over {len(IFEVAL_TASKS)} test prompts).',
        'tab:ifeval-genroc'
    )

    doc = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{caption}

\title{Semi-Supervised Training Results}
\author{}
\date{}

\begin{document}
\maketitle

\section{PlausibleQA}

""" + tables['pqa_corr'] + '\n\n' + tables['pqa_roc'] + r"""

\section{AmbigQA}

""" + tables['ambigqa_corr'] + '\n\n' + tables['ambigqa_roc'] + r"""

\section{Hypernym (In-Domain)}

""" + tables['hyp_id_corr'] + '\n\n' + tables['hyp_id_roc'] + r"""

\section{Hypernym (Out-of-Domain)}

""" + tables['hyp_ood_corr'] + '\n\n' + tables['hyp_ood_roc'] + r"""

\section{IFEval}

""" + tables['ifeval_corr'] + '\n\n' + tables['ifeval_roc'] + r"""

\end{document}
"""

    return doc


if __name__ == '__main__':
    doc = generate_all_tables()
    output_path = Path(__file__).resolve().parent.parent / 'semisupervised.tex'
    with open(output_path, 'w') as f:
        f.write(doc)
    print(f"\nWrote {output_path}")
