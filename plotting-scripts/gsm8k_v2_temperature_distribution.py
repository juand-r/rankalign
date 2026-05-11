"""
Heatmaps of temperature co-occurrence with (model, strategy) for gsm8k-v1
candidates (gsm8k-v2 version of humaneval_v1_temperature_distribution.py).
Helps see whether temperature is confounded with the other factors.

For each of (model, strategy) we draw two heatmaps:
  - left: raw counts
  - right: row-normalized (within-row %) so you can compare distributions
    across rows directly

Output: output-metrics/gsm8k_v2_temperature_distribution.png
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC_GLOB = 'data/gsm8k/v2/test/gsm8k_test_*.csv'
OUT = 'output-metrics/gsm8k_v2_temperature_distribution.png'


def load_src() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob(SRC_GLOB))],
        ignore_index=True,
    )
    df['model_canon'] = df['model'].str.replace('GPT-4o', 'gpt-4o', regex=False)
    df['temp_str'] = df['temperature'].apply(
        lambda v: 'unknown' if pd.isna(v) else f'{v:g}'
    )
    return df


def _temp_sort_key(label: str) -> tuple[int, float]:
    if label == 'unknown':
        return (1, float('inf'))
    return (0, float(label))


def crosstab_counts(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    ct = pd.crosstab(df[row], df[col])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    ct = ct[sorted(ct.columns, key=_temp_sort_key)]
    return ct


def plot_heatmap(ax, ct: pd.DataFrame, title: str, normalize_row: bool,
                 fmt: str, cmap: str, cbar_label: str):
    if normalize_row:
        row_sums = ct.sum(axis=1).replace(0, np.nan)
        data = (ct.div(row_sums, axis=0) * 100).fillna(0)
    else:
        data = ct

    im = ax.imshow(data.values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=0, fontsize=9)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([f'{idx}  (n={ct.loc[idx].sum()})' for idx in ct.index], fontsize=9)
    ax.set_xlabel('temperature', fontsize=10)
    ax.set_title(title, fontsize=11)

    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v_show = data.iloc[i, j]
            v_color = data.iloc[i, j]
            if normalize_row:
                cell = '' if v_show < 0.5 else f'{v_show:.0f}%'
            else:
                cell = '' if v_show == 0 else f'{int(v_show)}'
            if cell:
                vmax = data.values.max() if data.values.max() > 0 else 1
                color = 'white' if v_color > 0.55 * vmax else 'black'
                ax.text(j, i, cell, ha='center', va='center', fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def main():
    df = load_src()
    print(f'{len(df)} candidates over {df.task_id.nunique()} problems')
    print(f"  models: {df['model_canon'].nunique()},  strategies: {df['strategy'].nunique()},  temperatures: {df['temperature'].nunique()}")

    ct_model = crosstab_counts(df, 'model_canon', 'temp_str')
    ct_strat = crosstab_counts(df, 'strategy', 'temp_str')

    print('\nmodel x temperature (counts):')
    print(ct_model.to_string())
    print('\nstrategy x temperature (counts):')
    print(ct_strat.to_string())

    fig = plt.figure(figsize=(15, max(10, 0.45 * (len(ct_model) + len(ct_strat)) + 4)))
    gs = fig.add_gridspec(2, 2, height_ratios=[len(ct_model), len(ct_strat)],
                          hspace=0.35, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    plot_heatmap(ax0, ct_model, 'Model × Temperature  (counts)',
                 normalize_row=False, fmt='d', cmap='Blues', cbar_label='count')
    ax0.set_ylabel('model', fontsize=10)

    ax1 = fig.add_subplot(gs[0, 1])
    plot_heatmap(ax1, ct_model, 'Model × Temperature  (% within-model)',
                 normalize_row=True, fmt='.0f', cmap='Reds', cbar_label='% of model row')

    ax2 = fig.add_subplot(gs[1, 0])
    plot_heatmap(ax2, ct_strat, 'Strategy × Temperature  (counts)',
                 normalize_row=False, fmt='d', cmap='Blues', cbar_label='count')
    ax2.set_ylabel('strategy', fontsize=10)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax3, ct_strat, 'Strategy × Temperature  (% within-strategy)',
                 normalize_row=True, fmt='.0f', cmap='Reds', cbar_label='% of strategy row')

    fig.suptitle(f'gsm8k-v1 candidate distribution: temperature vs (model, strategy)\n'
                 f'{len(df)} candidates, {df.task_id.nunique()} problems',
                 fontsize=12)
    plt.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
