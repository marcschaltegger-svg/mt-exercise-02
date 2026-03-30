import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# CLI

parser = argparse.ArgumentParser(description='Visualise perplexity log files from main.py')
parser.add_argument('log_files', nargs='+', metavar='TSV',
                    help='One or more TSV log files produced by main.py --log-file')
parser.add_argument('--out-dir', type=str, default='results',
                    help='Directory where tables (CSV) and plots (PNG) are saved (default: results/)')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


# load log files

frames = []
for path in args.log_files:
    try:
        df = pd.read_csv(path, sep='\t')
        frames.append(df)
        print(f'Loaded {path}  ({len(df)} rows)')
    except Exception as e:
        print(f'WARNING: could not read {path}: {e}', file=sys.stderr)

if not frames:
    sys.exit('No log files could be loaded. Exiting.')

all_data = pd.concat(frames, ignore_index=True)


all_data['run_label'] = all_data.apply(
    lambda r: f'{r["model"]}  dropout={r["dropout"]}', axis=1)



def make_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(index='epoch', columns='run_label',
                           values=value_col, aggfunc='mean')
    pivot.index.name = 'Epoch'
    return pivot

#summary tables (CSV)

for metric, col in [('train', 'train_ppl'), ('validation', 'val_ppl')]:
    pivot = make_pivot(all_data, col)
    out_path = os.path.join(args.out_dir, f'table_{metric}_ppl.csv')
    pivot.to_csv(out_path, float_format='%.2f')
    print(f'Saved table  → {out_path}')
    print(pivot.to_string(), '\n')


#line charts

STYLE = {
    'figure.figsize': (9, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'font.size': 11,
}

def plot_metric(df: pd.DataFrame, col: str, title: str, ylabel: str,
                out_path: str) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        pivot = make_pivot(df, col)
        for run_label in pivot.columns:
            ax.plot(pivot.index, pivot[run_label], marker='o', markersize=3,
                    linewidth=1.8, label=run_label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, loc='upper right')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f'Saved plot   → {out_path}')


plot_metric(
    all_data, 'train_ppl',
    title='Training Perplexity per Epoch',
    ylabel='Perplexity',
    out_path=os.path.join(args.out_dir, 'plot_train_ppl.png'),
)

plot_metric(
    all_data, 'val_ppl',
    title='Validation Perplexity per Epoch',
    ylabel='Perplexity',
    out_path=os.path.join(args.out_dir, 'plot_val_ppl.png'),
)

print('\nDone. All outputs saved to:', os.path.abspath(args.out_dir))
