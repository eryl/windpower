import argparse
from csv import DictReader
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    parser.add_argument('performance_files', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--performance-metric', help="What column of the performance datasets to use", default='mean_absolute_error')
    args = parser.parse_args()

    models = set()

    model_pair_performance = defaultdict(list)
    for performance_file in args.performance_files:
        performance = pd.read_csv(performance_file)
        site_ids = set(performance['site_id'])
        model_a, model_b = list(sorted(set(performance['nwp_model'])))
        models.add(model_a)
        models.add(model_b)
        for site_id in site_ids:
            site_performance = performance[performance['site_id'] == site_id]
            fold_ids = set(site_performance['outer_fold_id'])
            for fold_id in fold_ids:
                fold_performances = site_performance[site_performance['outer_fold_id'] == fold_id]
                model_a_performance = fold_performances[fold_performances['nwp_model'] == model_a][args.performance_metric]
                model_b_performance = fold_performances[fold_performances['nwp_model'] == model_b][args.performance_metric]
                model_pair_performance[(model_a, model_b)].append((1 - model_a_performance.mean()/model_b_performance.mean()))
                model_pair_performance[(model_b, model_a)].append((1 - model_b_performance.mean()/model_a_performance.mean()))
                #model_pair_performance[(model_a, model_b)].append(model_a_performance.mean() - model_b_performance.mean())
                #model_pair_performance[(model_b, model_a)].append(model_b_performance.mean() - model_a_performance.mean())

    make_grid_histograms(models, model_pair_performance, args.performance_metric, args.output_dir)
    #make_box_plots(models, model_pair_performance, args.performance_metric, args.output_dir)
    plt.show()

def make_grid_histograms(models, model_pair_performance, performance_metric, output_dir=None):
    fig, subplots = plt.subplots(len(models), len(models), squeeze=False, sharex='col', sharey='row')
    cmap = plt.get_cmap('viridis')
    model_a_better = cmap(0.75)
    model_b_better = cmap(0.25)
    inconclusive = cmap(0.)
    for i, model_a in enumerate(sorted(models)):
        for j, model_b in enumerate(sorted(models)):
            ax = subplots[i, j]

            if j == 0:
                # First column, set y label
                ax.set_ylabel(model_a, fontsize=10)

            if i == len(models) - 1:
                ## Final row, add x labels
                ax.set_xlabel(model_b, fontsize=10)

            if model_a == model_b:
                #x_axis = ax.get_xaxis()
                #x_axis.set_ticks([])
                #x_axis.set_ticklabels([])
                #y_axis = ax.get_yaxis()
                #y_axis.set_ticks([])
                #y_axis.set_ticklabels([])
                continue
            performance = model_pair_performance[(model_a, model_b)]
            dist_plot = sns.distplot(performance, bins=40, kde=False, ax=ax)

            for rectangle in dist_plot.containers[0]:  # Containers has one member, a BarContainer which in turn contains all the bars
                x = rectangle.xy[0]
                rectangle.set_alpha(.7)
                if x > 0:
                    rectangle.set_facecolor(model_a_better)
                elif x + rectangle._width < 0:
                    rectangle.set_facecolor(model_b_better)
                else:
                    rectangle.set_facecolor(inconclusive)

    fig.text(0.5, 0.05, 'Model B', ha='center')
    fig.text(0.04, 0.5, 'Model A', va='center', rotation='vertical')
    fig.suptitle(f"Pairwise skill score, $1 - \\frac{{Model A}}{{Model B}}$, of {performance_metric}."
                 f" \nValues higher than 0 favors model_name A, lower favors model_name B")
    legend_elements = [Patch(facecolor=model_a_better, edgecolor='white',
                         label='Model A is better', alpha=.7),
                       Patch(facecolor=model_b_better, edgecolor='white',
                             label='Model B is better', alpha=.7),
                       Patch(facecolor=inconclusive, edgecolor='white',
                             label='Inconclusive', alpha=.7)
                       ]
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.8, 0.7, 0.15, 0.1), loc='upper right',
           ncol=1, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.926,
                        bottom=0.155,
                        left=0.1,
                        right=0.75,
                        hspace=0.154,
                        wspace=0.086)
    if output_dir is not None:

        save_path = output_dir / f'pairwise_nwp_comparison.png'
        plt.savefig(save_path)


def make_box_plots(models, model_pair_performance, performance_metric, output_dir=None):
    fig = plt.figure()
    long_form_data = {'models': [], 'skill score': []}
    for model_a, model_b in itertools.combinations(sorted(models), 2):
        for performance in model_pair_performance[(model_a, model_b)]:
            #label_a = '\\textnormal{' + model_a + '}'
            #label_a = model_a
            #label_b = '\\textnormal{' + model_b + '}'
            #label_b = model_b
            #label = '$\\frac{' + label_a + '}{' + label_b + '}$'
            label = f'Model A:{model_a}\nModel B: {model_b}'
            long_form_data['models'].append(label)
            long_form_data['skill score'].append(performance)
    df = pd.DataFrame(long_form_data)
    sns.boxplot(data=df, x='models', y='skill score')
    plt.ylabel('Skill score {}'.format(performance_metric.replace('_', ' ')))
    ax = plt.gca()
    # ax.annotate('Model A is better',  xy=(0, 1), xycoords=ax.get_yaxis_transform(),
    #                xytext=(-5,0), textcoords="offset points", ha="right", va="center")
    # ax.annotate('Model B is better', xy=(0, 0), xycoords=ax.get_yaxis_transform(),
    #             xytext=(-5, 0), textcoords="offset points", ha="right", va="center")

    fig.suptitle(f"Pairwise skill score, $1 - \\frac{{Model A}}{{Model B}}$, of {performance_metric}."
                 f" \nValues higher than 0 favors model_name A, lower favors model_name B")


if __name__ == '__main__':
    main()