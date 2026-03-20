#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the last N rows of eval_results.csv to a LaTeX table.

Usage:
    python scripts/csv_to_latex.py <N> <output_filename.tex>
    
Example:
    python scripts/csv_to_latex.py 10 my_table.tex
"""

import argparse
import csv
import re
import os


def parse_model_path(model_path):
    """
    Extract model info from path like:
    ../models/v5-google--gemma-2-2b-delta2.5-epoch2--hypernym-car-all--g2d--random--alpha1.0--typcorr
    
    Returns dict with: model, delta, epoch, method, typcorr
    """
    result = {
        'model': '',
        'delta': '',
        'epoch': '',
        'method': '',
        'typcorr': 'No'
    }
    
    # Check for typcorr
    if 'typcorr' in model_path:
        result['typcorr'] = 'Yes'
    
    # Extract model name (e.g., gemma-2-2b, Llama-3.1-8B)
    # Pattern: after org-- and before -delta
    model_match = re.search(r'--([a-zA-Z][\w.-]+)-delta', model_path)
    if model_match:
        result['model'] = model_match.group(1)
    else:
        # Try alternative: direct model name like google/gemma-2-2b
        alt_match = re.search(r'([^/]+/[\w.-]+)$', model_path.rstrip('/'))
        if alt_match:
            result['model'] = alt_match.group(1)
        else:
            result['model'] = model_path.split('/')[-1][:30]  # Fallback: truncated path
    
    # Strip common org prefixes like "google/", "meta-llama/", etc.
    result['model'] = re.sub(r'^(google|meta-llama|mistralai|microsoft)/', '', result['model'])
    
    # Extract delta
    delta_match = re.search(r'-delta([\d.]+)', model_path)
    if delta_match:
        result['delta'] = delta_match.group(1)
    
    # Extract epoch
    epoch_match = re.search(r'-epoch(\d+)', model_path)
    if epoch_match:
        result['epoch'] = epoch_match.group(1)
    
    # Extract method (g2d or d2g)
    method_match = re.search(r'--(g2d|d2g)--', model_path)
    if method_match:
        result['method'] = method_match.group(1)
    
    return result


def format_value(value, scale=100, decimals=1):
    """Format a numeric value: scale by 100 and round to 1 decimal."""
    if value == '' or value == 'nan':
        return '--'
    try:
        num = float(value)
        if num != num:  # NaN check
            return '--'
        scaled = num * scale
        fmt = "{:." + str(decimals) + "f}"
        return fmt.format(scaled)
    except (ValueError, TypeError):
        return '--'


def generate_latex_table(rows, header):
    """Generate LaTeX table from rows."""
    
    # Column mapping from CSV to LaTeX
    csv_cols = {
        'task': header.index('task'),
        'model': header.index('model'),
        'spear_all': header.index('spear_all'),
        'spear_pos': header.index('spear_pos'),
        'spear_neg': header.index('spear_neg'),
        'disc_acc': header.index('disc_acc'),
        'disc_roc': header.index('disc_roc'),
        'gen_acc_100': header.index(' gen_acc_100'),  # Note: has leading space
        'gen_mrr_pos': header.index('gen_mrr_pos'),
        'gen_mrr_neg': header.index(' gen_mrr_neg'),  # Note: has leading space
        'gen_roc': header.index('gen_roc'),
    }
    
    # gen_acc_100_dataset might have leading space
    try:
        csv_cols['gen_acc_100_dataset'] = header.index('gen_acc_100_dataset')
    except ValueError:
        try:
            csv_cols['gen_acc_100_dataset'] = header.index(' gen_acc_100_dataset')
        except ValueError:
            csv_cols['gen_acc_100_dataset'] = None
    
    # Group rows by task
    task_groups = {}
    for row in rows:
        task = row[csv_cols['task']]
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(row)
    
    # Build table content
    table_rows = []
    tasks = list(task_groups.keys())
    
    for task_idx, task in enumerate(tasks):
        task_rows = task_groups[task]
        num_rows = len(task_rows)
        
        for i, row in enumerate(task_rows):
            # Parse model path
            model_info = parse_model_path(row[csv_cols['model']])
            
            # Format method: g2d -> G, d2g -> V
            method_display = '--'
            if model_info['method']:
                last_char = model_info['method'][-1].upper()
                method_display = 'V' if last_char == 'D' else last_char
            
            # Format typcorr: Yes in red
            typcorr_display = model_info['typcorr']
            if typcorr_display == 'Yes':
                typcorr_display = r'\textcolor{red}{Yes}'
            
            # Build row values
            values = [
                model_info['model'],
                model_info['delta'] if model_info['delta'] else '--',
                model_info['epoch'] if model_info['epoch'] else '--',
                method_display,
                typcorr_display,
                format_value(row[csv_cols['spear_all']]),
                format_value(row[csv_cols['spear_pos']]),
                format_value(row[csv_cols['spear_neg']]),
                format_value(row[csv_cols['disc_acc']]),
                format_value(row[csv_cols['disc_roc']]),
                format_value(row[csv_cols['gen_mrr_pos']]),
                format_value(row[csv_cols['gen_mrr_neg']]),
                format_value(row[csv_cols['gen_roc']]),
            ]
            
            # Add dataset-constrained accuracy if available
            if csv_cols['gen_acc_100_dataset'] is not None:
                values.append(format_value(row[csv_cols['gen_acc_100_dataset']]))
            else:
                values.append('--')
            
            # Format row with task multirow
            if i == 0:
                task_cell = f"\\multirow{{{num_rows}}}{{*}}{{{task}}}"
            else:
                task_cell = ""
            
            row_str = f"          {task_cell} & " + " & ".join(values) + " \\\\"
            table_rows.append(row_str)
        
        # Add midrule between task groups (but not after the last one)
        if task_idx < len(tasks) - 1:
            table_rows.append("\\midrule\\midrule")
    
    # Build full table
    latex = r"""\begin{table}[t!]
\small
\centering
\vspace{-1em}
\begin{tabular}{ll|cccc|ccc|cc|cccc}
\toprule
Task & Model & $\delta$ & Ep & Meth & TC & $\rho$-all & $\rho$-pos & $\rho$-neg & Acc & ROC & MRR-P & MRR-N & ROC-g & A-d@100 \\
\midrule
"""
    latex += "\n".join(table_rows)
    latex += r"""
\bottomrule
\end{tabular}
\caption{Automatically generated latex table.}
\label{tab:auto-generated}
\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Convert last N rows of eval_results.csv to LaTeX table')
    parser.add_argument('n', type=int, help='Number of rows to include (from the end)')
    parser.add_argument('output_filename', type=str, help='Output filename (e.g., my_table.tex)')
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'outputs', 'eval_results.csv')
    output_path = os.path.join(project_root, 'outputs', args.output_filename)
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    
    header = all_rows[0]
    data_rows = all_rows[1:]  # Skip header
    
    # Get last N rows
    if args.n > len(data_rows):
        print(f"Warning: Requested {args.n} rows but only {len(data_rows)} available. Using all rows.")
        selected_rows = data_rows
    else:
        selected_rows = data_rows[-args.n:]
    
    # Generate LaTeX
    latex_content = generate_latex_table(selected_rows, header)
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table written to: {output_path}")
    print(f"Included {len(selected_rows)} rows from {len(set(row[header.index('task')] for row in selected_rows))} task(s)")


if __name__ == '__main__':
    main()

