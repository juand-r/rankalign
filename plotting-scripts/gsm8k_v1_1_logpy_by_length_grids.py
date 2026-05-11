"""
Mean typicality-term log-prob by completion length, faceted by subsets of the
data, for gsm8k-v1 (gsm8k-v1.1 version of humaneval_v1_logpy_by_length_grids.py).
Produces three separate figure grids:
  1. split by candidate-generator model
  2. split by generation strategy
  3. split by sampling temperature

The "typicality term" is gen_score - gen_score_typcorr:
  --mode self  -> log P(y)         (unconditional self-typicality)
  --mode neg   -> log P(y | x_neg) (typicality under negated prompt)

Outputs (with {mode} = self or neg):
  output-metrics/gsm8k_v1.1_1_{mode}_logptyp_by_length__by_model.png
  output-metrics/gsm8k_v1.1_1_{mode}_logptyp_by_length__by_strategy.png
  output-metrics/gsm8k_v1.1_1_{mode}_logptyp_by_length__by_temperature.png
"""
import argparse, glob, math, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

SRC_GLOB = 'data/gsm8k/v1.1/gsm8k_test_*.csv'
OUT_DIR = 'output-metrics'
MIN_PER_BIN = 4
MIN_PANEL_N = 30

# gsm8k-v1 score CSVs do NOT populate the `problem_name` column (unlike
# humaneval-v1); recover problem_id from the score filename instead.
PID_RE = re.compile(r'gsm8k-v1.1-gsm8k_test_(\d+)_test')


def make_mode_labels(mode: str, scores_dir: str, model_glob: str) -> dict:
    common = {
        'self': dict(
            ylabel='mean log P(y)',
            title='mean unconditional log P(y)',
        ),
        'neg': dict(
            ylabel='mean log P(y | x_neg)',
            title='mean log P(y | x_neg)  (under negated prompt)',
        ),
    }[mode]
    # Literal `_gsm8k-v1` after the model substring keeps gemma-2-2b separate
    # from gemma-2-2b-it.
    common['glob'] = f'{scores_dir}/scores_{mode}-{model_glob}_gsm8k-v1.1-*.csv'
    return common


def load_merged(mode: str, mode_labels: dict) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(mode_labels['glob'])):
        m = PID_RE.search(os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f)
        d['problem_id'] = int(m.group(1))
        parts.append(d)
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    src = pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob(SRC_GLOB))],
        ignore_index=True,
    )
    src['problem_id'] = src['task_id'].str.extract(r'gsm8k_test_(\d+)').astype(int)
    src['solution_preview'] = src['answer'].str[:200].str.replace('\n', '\\n', regex=False)

    keys = ['problem_id', 'solution_preview', 'strategy']
    ev = ev.sort_values(keys).reset_index(drop=True)
    src = src.sort_values(keys).reset_index(drop=True)
    src['seq'] = src.groupby(keys).cumcount()
    ev['seq'] = ev.groupby(keys).cumcount()

    merged = ev.merge(
        src[['problem_id', 'solution_preview', 'strategy', 'seq', 'model', 'temperature']],
        on=['problem_id', 'solution_preview', 'strategy', 'seq'],
        how='left',
    )
    merged['label'] = (merged['correct'] == 'yes').astype(int)
    merged['logpy'] = merged['gen_score'] - merged['gen_score_typcorr']
    n_unmatched = merged['model'].isna().sum()
    if n_unmatched:
        print(f"WARN: {n_unmatched} rows did not match source; dropping them")
        merged = merged.dropna(subset=['model'])
    return merged


def plot_panel(ax, sub_df: pd.DataFrame, bins: np.ndarray, title: str,
               ylabel: str = 'mean log P(y)',
               show_xlabel: bool = True, show_ylabel: bool = True):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    correct = sub_df[sub_df.label == 1]
    incorrect = sub_df[sub_df.label == 0]

    if len(correct) >= 4:
        c_means, _, _ = binned_statistic(correct.num_tokens, correct.logpy, statistic='mean', bins=bins)
        c_counts, _, _ = binned_statistic(correct.num_tokens, correct.logpy, statistic='count', bins=bins)
        c_mask = c_counts >= MIN_PER_BIN
        if c_mask.any():
            ax.plot(bin_centers[c_mask], c_means[c_mask], 'g-o', markersize=5,
                    linewidth=1.8, label=f'correct (n={len(correct)})')

    if len(incorrect) >= 4:
        ic_means, _, _ = binned_statistic(incorrect.num_tokens, incorrect.logpy, statistic='mean', bins=bins)
        ic_counts, _, _ = binned_statistic(incorrect.num_tokens, incorrect.logpy, statistic='count', bins=bins)
        ic_mask = ic_counts >= MIN_PER_BIN
        if ic_mask.any():
            ax.plot(bin_centers[ic_mask], ic_means[ic_mask], 'r-o', markersize=5,
                    linewidth=1.8, label=f'incorrect (n={len(incorrect)})')

    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel('num_tokens', fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')


def make_grid(df: pd.DataFrame, facet_col: str, out_path: str, suptitle: str,
              ncols: int, ylabel: str = 'mean log P(y)',
              max_facets: int | None = None, sort_by: str = 'count'):
    ntok_max = int(df.num_tokens.quantile(0.98))
    bin_width = max(50, ntok_max // 10)
    bins = np.arange(0, ntok_max + bin_width, bin_width)
    ymin = df['logpy'].quantile(0.02)
    ymax = df['logpy'].quantile(0.98)

    counts = df[facet_col].value_counts()
    if sort_by == 'value':
        keep = sorted([k for k, v in counts.items() if v >= MIN_PANEL_N])
    else:
        keep = [k for k, v in counts.items() if v >= MIN_PANEL_N]
        if max_facets:
            keep = keep[:max_facets]

    panels = [('ALL', df)] + [(k, df[df[facet_col] == k]) for k in keep]
    n = len(panels)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for idx, (lab, sub) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        title = f'ALL (n={len(sub)})' if lab == 'ALL' else f'{lab}  (n={len(sub)})'
        plot_panel(ax, sub, bins, title, ylabel=ylabel,
                   show_xlabel=(r == nrows - 1),
                   show_ylabel=(c == 0))
        ax.set_ylim(ymin, ymax)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['self', 'neg'], default='self')
    parser.add_argument('--eval-model', default=None,
                        help='Substring matched against eval-model in score filenames.')
    parser.add_argument('--scores-dir', default='outputs',
                        help='Directory holding score CSVs (default outputs).')
    args = parser.parse_args()

    eval_model = args.eval_model
    model_glob = f'*{eval_model}' if eval_model else '*'
    model_tag = f'__{eval_model}' if eval_model else ''
    model_title = eval_model if eval_model else 'all eval models'

    mode_labels = make_mode_labels(args.mode, args.scores_dir, model_glob)
    df = load_merged(args.mode, mode_labels)
    print(f'[{args.mode}] Loaded {len(df)} merged rows ({df.problem_id.nunique()} problems, '
          f'{df.label.sum()} correct / {(df.label == 0).sum()} incorrect)')

    df['model_canon'] = df['model'].str.replace('GPT-4o', 'gpt-4o', regex=False)

    ylabel = mode_labels['ylabel']
    base_title = (
        f'gsm8k-v1: {mode_labels["title"]} by completion length\n'
        f'(eval model: {model_title}, --{args.mode}-typcorr)'
    )

    out_prefix = f'{OUT_DIR}/gsm8k_v1.1_1_{args.mode}_logptyp_by_length'

    make_grid(df, 'model_canon',
              f'{out_prefix}__by_model{model_tag}.png',
              suptitle=f'{base_title}\nfaceted by candidate-generator model',
              ncols=4, ylabel=ylabel, max_facets=15)

    make_grid(df, 'strategy',
              f'{out_prefix}__by_strategy{model_tag}.png',
              suptitle=f'{base_title}\nfaceted by generation strategy',
              ncols=4, ylabel=ylabel)

    make_grid(df, 'temperature',
              f'{out_prefix}__by_temperature{model_tag}.png',
              suptitle=f'{base_title}\nfaceted by sampling temperature',
              ncols=4, ylabel=ylabel, sort_by='value')


if __name__ == '__main__':
    main()
