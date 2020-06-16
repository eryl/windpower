import argparse
from csv import DictReader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
sns.set(color_codes=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('performance_files', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    performance = []
    for f in args.performance_files:
        with open(f) as fp:
            performance.extend(list(DictReader(fp)))
    cols = set()
    for p in performance:
        cols.update(p.keys())

    data_dicts = defaultdict(list)
    for p in performance:
        for c in cols:
            v = p.get(c, None)
            data_dicts[c].append(v)

    df = pd.DataFrame(data_dicts)

    nwp_models = df['nwp_model'].unique()
    mae = df['mean_absolute_error']
    df['mean_absolute_error'] = pd.to_numeric(df['mean_absolute_error'], errors='coerce')
    df = df.dropna(axis=0, how='any', subset=['mean_absolute_error'])
    mae_min = df['mean_absolute_error'].min()
    mae_max = df['mean_absolute_error'].max()
    models = df['model'].unique()
    fig, axes = plt.subplots(len(nwp_models), len(models), sharex='col', squeeze=False, sharey='all')
    for i, nwp_model in enumerate(sorted(nwp_models)):
        for j, model in enumerate(sorted(models)):
            ax = axes[i, j]
            model_df = df.loc[np.logical_and(df['nwp_model'] == nwp_model, df['model'] == model)]
            try:
                sns.boxplot(data=model_df, x='fold_id', y='mean_absolute_error', ax=ax)
            except ValueError:
                continue
            ax.set_title(nwp_model)
            if i == len(nwp_models) - 1:
                ax.set_xlabel('Test fold id')
            else:
                ax.set_xlabel('')
            if i == 0:
                ax.text(0.5, 1.3, model, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontdict=dict(size=18))
            if j == 0:
                ax.set_ylabel('Mean absolute error')
            else:
                ax.set_ylabel('')

    if args.output_dir is not None:

        save_path = args.output_dir / f'nwp_compare.png'
        plt.savefig(save_path)
    plt.show()



if __name__ == '__main__':
    main()