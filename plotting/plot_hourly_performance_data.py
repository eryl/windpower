import argparse
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
    parser.add_argument('performance_file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    performance = pd.read_csv(args.performance_file)
    fig = plt.figure(figsize=(12, 8))
    sns.kdeplot(data=performance['model_a_mae'], label='Neural networks')
    sns.kdeplot(data=performance['model_b_mae'], label='Gradient boosted trees')
    plt.xlabel('Mean absolute error')
    plt.legend()
    plt.tight_layout()
    fig.savefig('neural_net_vs_gradient_boosted_kde_plots.pdf')

    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=performance, x='lead_time', y='model_a_mae', label='Neural networks', ci=99)
    sns.lineplot(data=performance, x='lead_time', y='model_b_mae', label='Gradient boosted trees', ci=99)
    plt.xlabel("Lead time")
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.tight_layout()
    fig.savefig('neural_net_vs_gradient_boosted_tree_lead_time.pdf')
    plt.show()


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
