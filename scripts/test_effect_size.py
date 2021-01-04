import argparse
import csv
from csv import DictReader
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats


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
    parser.add_argument('--output-file', type=Path)
    parser.add_argument('--performance-metric', help="What column of the performance datasets to use", default='mean_absolute_error')
    args = parser.parse_args()

    performance_files = []
    for d in args.performance_files:
        if d.is_dir():
            performance_files.extend(d.glob('*.csv'))
        else:
            performance_files.append(d)
    models = set()

    model_pair_performance = defaultdict(lambda: defaultdict(list))
    for performance_file in performance_files:
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
                model_pair_performance[frozenset((model_a, model_b))][model_a].append(model_a_performance.mean())
                model_pair_performance[frozenset((model_a, model_b))][model_b].append(model_b_performance.mean())

    model_as = []
    model_bs = []
    mean_a = []
    mean_b = []
    std_a = []
    std_b = []
    a_higher_than_b = []
    c_ds = []

    if args.output_file is not None:
        fieldnames = ['model_a', 'model_b', 'mean_a', 'mean_b', 'std_a', 'std_b', 'n_a', 'n_b', 'cliffs_d', 'cohens_d']
        with open(args.output_file, 'w') as fp:
            csv_writer = csv.DictWriter(fp, fieldnames=fieldnames)
            csv_writer.writeheader()
            for (model_a, model_b), performances in model_pair_performance.items():
                a = performances[model_a]
                b = performances[model_b]

                # statistic, pvalue = scipy.stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy='propagate')
                # print(f"Model a: {model_a}, model b: {model_b}: t-statistic: {statistic}, p-value: {pvalue}")
                a_w_1 = nonparametric_effect_size(a, b)
                a_w_2 = nonparametric_effect_size(b, a)
                print(f"Probability that {model_a} performs worse than {model_b}: {a_w_1}")
                print(f"Probability that {model_b} performs worse than {model_a}: {a_w_2}")

                c_d = cohens_d(a,b)
                print(f"Cohen's d for {model_a}, {model_b}: {c_d}")

                csv_writer.writerow({
                    'model_a': model_a,
                    'model_b': model_b,
                    'mean_a': np.mean(a),
                    'mean_b': np.mean(b),
                    'std_a': np.std(a),
                    'std_b': np.std(b),
                    'n_a': len(a),
                    'n_b': len(b),
                    'cliffs_d': a_w_1,
                    'cohens_d': c_d
                })


def paired_difference(model_pair_performance):
    for (model_a, model_b), skills in model_pair_performance.items():

        p = (np.array(skills) > 0).sum()/len(skills)
        print(f"Probability that {model_a} outperforms {model_b}: {p}")



def cohens_d(a, b):
    """Calculate Cohen's d for the two datasets"""
    n1 = len(a)
    n2 = len(b)
    s1_sqr = np.var(a)
    s2_sqr = np.var(b)

    s = np.sqrt(((n1-1)*s1_sqr + (n2-1)*s2_sqr)/(n1+n2-2))
    d = (np.mean(a) - np.mean(b))/s
    return np.abs(d)


def nonparametric_effect_size(a, b):
    """Implementation of Cliff's delta d, but ties add a count of 0.5"""

    a = np.array(a)
    b = np.array(b)
    total_count = 0
    for x in a:
        higher_count = (x > b).sum() + (x == b).sum()/2
        total_count += higher_count
    A_w = total_count/(len(a)*len(b))

    return A_w


if __name__ == '__main__':
    #print(nonparametric_effect_size([5, 7, 6, 5], [3, 4, 5, 3]))
    main()