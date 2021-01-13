import argparse
import csv
from csv import DictReader
import itertools
#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.patches import Patch
import scipy.stats
from tqdm import tqdm

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
#matplotlib.verbose.level = 'debug-annoying'
#params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
#plt.rcParams.update(params)

import pandas as pd
#import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
#sns.set(color_codes=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('performance_files', type=Path, nargs='+')
    parser.add_argument('--output-file', type=Path)
    parser.add_argument('--performance-metric', help="What column of the performance datasets to use", default='mean_absolute_error')
    parser.add_argument('--bootstrap-n', type=int, default=10000)
    parser.add_argument('--ci', type=int, default=99)
    args = parser.parse_args()

    performance_files = []
    for d in args.performance_files:
        if d.is_dir():
            performance_files.extend(d.glob('*.csv'))
        else:
            performance_files.append(d)
    models = set()

    model_pair_performance = defaultdict(lambda: defaultdict(list))
    for performance_file in tqdm(performance_files, desc="Reading performance data"):
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


    if args.output_file is not None:
        fieldnames = ['model_a', 'model_b', 'mean_a', 'mean_b', 'std_a', 'std_b', 'n_a', 'n_b', 'cliffs_d', 'cohens_d', 'cliffs_d_bootstrap_low',
                    'cliffs_d_bootstrap_high',
                    'cohens_d_bootstrap_low',
                    'cohens_d_bootstrap_high',]
        with open(args.output_file, 'w') as fp:
            csv_writer = csv.DictWriter(fp, fieldnames=fieldnames)
            csv_writer.writeheader()
            for (model_a, model_b), performances in tqdm(model_pair_performance.items(), desc="Calculating effect size", total=len(model_pair_performance)):
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

                bs_cohens_d, bs_cohens_d_low, bs_cohens_d_high = bootstrapped_ci(a, b, cohens_d, n=args.bootstrap_n, ci=args.ci)
                bs_cliffs_d, bs_cliffs_d_low, bs_cliffs_d_high = bootstrapped_ci(a, b, nonparametric_effect_size, n=args.bootstrap_n, ci=args.ci)

                csv_writer.writerow({
                    'model_a': model_a,
                    'model_b': model_b,
                    'mean_a': '{:0.3f}'.format(np.mean(a)),
                    'mean_b': '{:0.3f}'.format(np.mean(b)),
                    'std_a': '{:0.3f}'.format(np.std(a)),
                    'std_b': '{:0.3f}'.format(np.std(b)),
                    'n_a': '{}'.format(len(a)),
                    'n_b': '{}'.format(len(b)),
                    'cliffs_d': bs_cliffs_d,
                    'cliffs_d_bootstrap_low': bs_cliffs_d_low,
                    'cliffs_d_bootstrap_high': bs_cliffs_d_high,
                    'cohens_d': bs_cohens_d,
                    'cohens_d_bootstrap_low': bs_cohens_d_low,
                    'cohens_d_bootstrap_high': bs_cohens_d_high,
                })


def paired_difference(model_pair_performance):
    for (model_a, model_b), skills in model_pair_performance.items():

        p = (np.array(skills) > 0).sum()/len(skills)
        print(f"Probability that {model_a} outperforms {model_b}: {p}")


def bootstrapped_ci(a, b, effect_size_f, n, ci):
    effect_sizes = [effect_size_f(bootstrap(a), bootstrap(b)) for i in range(n)]
    p = 50 - ci / 2, 50 + ci / 2
    low, high = np.percentile(effect_sizes, p)
    return np.mean(effect_sizes), low, high


def bootstrap(arr, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    resampled = rng.choice(arr, size=len(arr), replace=True)
    return resampled



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