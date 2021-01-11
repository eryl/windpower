import argparse
import re
from csv import DictReader
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats
from tqdm import tqdm

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
#matplotlib.verbose.level = 'debug-annoying'
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)


import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
sns.set(color_codes=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('importance_files', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, default=Path())
    parser.add_argument('--n-bootstraps', help="Number of bootstrap samples", type=int, default=10000)
    parser.add_argument('--ci', help="Confidence level in percents", type=int, default=99)
    args = parser.parse_args()

    performance_files = []
    for d in args.importance_files:
        if d.is_dir():
            performance_files.extend(d.glob('*.csv'))
        else:
            performance_files.append(d)

    models = set()
    nwp_feature_importance_files = defaultdict(list)
    pattern = r'.*_feature_importance_([\w\-\#]+)\.csv'
    for performance_file in performance_files:
        m = re.match(pattern, performance_file.name)
        if m is not None:
            nwp_model, = m.groups()
            nwp_feature_importance_files[nwp_model].append(performance_file)

    nwp_feature_importances = dict()
    for nwp_model, importance_files in tqdm(nwp_feature_importance_files.items(), desc='Merging importance files'):
        df = pd.concat([pd.read_csv(f) for f in importance_files])
        nwp_feature_importances[nwp_model] = df

    for nwp_model, df in tqdm(nwp_feature_importances.items(), desc="Calculating importance CIs"):
        #fig = plt.figure()
        #df2 = df.melt(var_name='cols', value_name='vals')
        #sns.lineplot(data=df2, x='cols', y='vals')
        mean, low, high = bootstrapped_ci(df.values, args.n_bootstraps, args.ci, axis=0)
        importance_ordered_indices = np.argsort(mean)[::-1]  # reversed order
        importance_columns = df.columns[importance_ordered_indices]
        ordered_means = mean[importance_ordered_indices]
        ordered_lows = low[importance_ordered_indices]
        ordered_highs = high[importance_ordered_indices]
        aggregated_df = pd.DataFrame(dict(feature=importance_columns, means=ordered_means, ci_low=ordered_lows, ci_high=ordered_highs))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        aggregated_df.to_csv(args.output_dir / f'aggregated_feature_importance_{nwp_model}.csv', index=False)
    plt.show()


def bootstrapped_ci(arr, n, ci, axis=0):
    bootstrap_samples = np.concatenate([np.mean(bootstrap(arr, axis=axis), axis=axis, keepdims=True) for i in range(n)], axis=axis)
    p = 50 - ci / 2, 50 + ci / 2
    low, high = np.percentile(bootstrap_samples, p, axis=axis)
    mean = np.mean(bootstrap_samples, axis=axis)
    return mean, low, high


def bootstrap(arr, rng=None, axis=0):
    if rng is None:
        rng = np.random.default_rng()
    n = arr.shape[axis]
    indices = rng.choice(n, size=n, replace=True)  # We reuse the indices for all columns, should be fine right?
    resampled = arr.take(indices, axis=axis)
    return resampled


def make_lead_time_plots(models, model_pair_performance, performance_metric, output_dir=None, plot_pairs=False):
    cmap = plt.get_cmap('viridis')
    models = sorted(models)
    model_colors = {model: cmap(i/len(models)) for i,model in enumerate(sorted(models))}
    for model in models:
        if '#' in model:
            model_colors['Merged models'] = model_colors[model]
            del model_colors[model]

    gathered_performance = []
    for (model_a, model_b), performances in tqdm(model_pair_performance.items(), total=len(model_pair_performance)):
        fig = plt.figure(figsize=(12,8))
        performance = pd.concat(performances)

        if "#" in model_a:
            performance.loc[performance['nwp_model']==model_a, 'nwp_model'] = 'Merged models'
            model_a = "Merged models"
        if "#" in model_b:
            performance.loc[performance['nwp_model']==model_b, 'nwp_model'] = 'Merged models'
            model_b = "Merged models"

        performance = performance.rename(columns={'nwp_model': 'NWP Model', 'lead_time': 'Lead time', 'mae': 'Normalized Mean Absolute Error'})

        gathered_performance.append(performance)
        if plot_pairs:
            sns.lineplot(data=performance, x='Lead time', y='Normalized Mean Absolute Error', hue='NWP Model', palette=model_colors, ci=99)

            fig.suptitle(f"Lead time comparisons of {model_a} vs {model_b}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.933,
                                bottom=0.083,
                                left=0.08,
                                right=0.985)

            if output_dir is not None:
                save_path = output_dir / f'lead_time_comparisons_{model_a}_{model_b}.png'
                plt.savefig(save_path)
                plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    performance = pd.concat(gathered_performance)
    sns.lineplot(data=performance, x='Lead time', y='Normalized Mean Absolute Error', hue='NWP Model',
                 palette=model_colors, legend=False, ci=99)
    legend_elements = [Patch(facecolor=color, edgecolor=color,label=model)
                       for model, color in sorted(model_colors.items())]
    plt.legend(handles=legend_elements, loc='upper left', ncol=1, title='NWP Model')

    plt.tight_layout()
    plt.subplots_adjust(top=0.933,
                        bottom=0.083,
                        left=0.08,
                        right=0.985)

    fig.suptitle(f"Lead time comparisons of all models")

    if output_dir is not None:
        save_path = output_dir / f'lead_time_comparisons_all_models.png'
        plt.savefig(save_path)
        plt.close(fig)


    # ax = plt.gca()
    # for nwp_model, performances in model_performances.items():
    #     collected_performances = defaultdict(list)
    #     for p in performances:
    #         for key, values in p.items():
    #             collected_performances[key].extend(values)
    #     df = pd.DataFrame(collected_performances)
    #     sns.lineplot(data=df, x='lead_time', y='mae', label=nwp_model)
    # plt.show()

    # for nwp_model, lead_time_performances in lead_time_vs_performance.items():
    #     x, ys = zip(*sorted(lead_time_performances.items()))
    #     # Each ys is a list of measurements
    #     means = np.array([np.mean(y) for y in ys])
    #     stds = np.array([np.std(y) for y in ys])
    #     CIs = 1.96 * stds/means
    #     plt.plot(x, means, label=nwp_model)
    #     plt.plot(x, means+CIs)
    #     plt.plot(x, means-CIs)
    #     plt.show()

if __name__ == '__main__':
    main()
