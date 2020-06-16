import argparse
import itertools
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

    nwp_models = set()
    model_weather_variable_performance = defaultdict(lambda: defaultdict(list))
    site_fold_model_performance = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for perf in performance:
        try:
            nwp_model = perf['nwp_model']
            nwp_models.add(nwp_model)
            mean_absolute_error = float(perf['mean_absolute_error'])
            r2 = float(perf['r_squared'])
            site_id = perf['site_id']
            fold_id = perf['fold_id']
            site_fold_model_performance[site_id][fold_id][nwp_model].append((mean_absolute_error, r2))
        except ValueError:
            continue

    nwp_models = list(sorted(nwp_models))
    fold_diffs = defaultdict(lambda: defaultdict(dict))
    for site_id, fold_model_performance in site_fold_model_performance.items():

        for fold_id, model_performance in fold_model_performance.items():
            model_mae = dict()
            model_r2 = dict()
            for nwp_model, performance in model_performance.items():
                mae, r2 = zip(*performance)
                model_mae[nwp_model] = np.mean(mae)
                model_r2[nwp_model] = np.mean(r2)
            for m1, m2 in itertools.combinations(nwp_models, 2):
                try:
                    mae_diff = model_mae[m1] - model_mae[m2]
                    r2_diff = model_r2[m1] - model_r2[m2]
                    fold_diffs[site_id][(m1, m2)][fold_id] = (mae_diff, r2_diff)
                except KeyError:
                    continue

    series_site_ids = []
    series_fold = []
    series_nwp_models = []
    series_mae_diff = []
    series_r2_diff = []

    for site_id, model_combination_diff in fold_diffs.items():
        for models, fold_diffs in model_combination_diff.items():
            for fold_id, (mae_diff, r2_diff) in fold_diffs.items():
                series_site_ids.append(site_id)
                series_fold.append(fold_id)
                series_nwp_models.append('-'.join(models))
                series_mae_diff.append(mae_diff)
                series_r2_diff.append(r2_diff)

    diff_df = pd.DataFrame({'site_id': series_site_ids,
                            'fold_id': series_fold,
                            'model_combination': series_nwp_models,
                            'mae_diff': series_mae_diff,
                            'r2_diff': series_r2_diff})

    sites = list(site_fold_model_performance.keys())
    n_sites_per_plot = 20
    n_plots = int(np.ceil(len(sites)/n_sites_per_plot))
    for i in range(n_plots):
        site_ids = sites[i*n_sites_per_plot: (i+1)*n_sites_per_plot]
        sns.catplot(x='site_id', y='mae_diff', kind="box", hue="model_combination", data=diff_df.loc[diff_df['site_id'].isin(site_ids)])
    #sns.catplot(x='site_id', y='r2_diff', kind="box", dodge=False, hue="model_combination", data=diff_df)
    #sns.boxplot(x='site_id', y='mae_diff', hue="model_combination", data=diff_df)
    plt.show()




if __name__ == '__main__':
    main()