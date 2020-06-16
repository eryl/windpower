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
    fig = plt.figure()
    for i, nwp_model in enumerate(sorted(nwp_models)):
        for j, model in enumerate(sorted(models)):
            model_df = df.loc[np.logical_and(df['nwp_model'] == nwp_model, df['model'] == model)]
            kde = sns.kdeplot(model_df['mean_absolute_error'], label=f'{model} - {nwp_model}',
                         #bins=50, hist_kws=dict(range=(mae_min, mae_max))
                        shade=True,
                        #shade_lowest=True
                        )
            color = kde.lines[-1].get_color()
            mean = model_df['mean_absolute_error'].mean()
            plt.vlines(mean, 0, 20, colors=color, linestyles='--', label=f"{nwp_model} mean, {mean:.03}")
    plt.xlabel("Mean absolute error")
    plt.suptitle("Kernel density estimations of error distributions")

    plt.legend()
    if args.output_dir is not None:
        save_path = args.output_dir / f'nwp_compare.png'
        plt.savefig(save_path)
    plt.show()




if __name__ == '__main__':
    main()