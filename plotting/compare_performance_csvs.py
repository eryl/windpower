import argparse
import pickle
from collections import defaultdict
import itertools

from pathlib import Path
import pandas as pd
import json
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare different performance files")
    parser.add_argument('performance_files', help="CSV files produced by gather_hp_data", type=Path, nargs='+')
    parser.add_argument('--outer-fold-idxs',
                        help="Which outer fold indices to compare. If none is given, "
                             "the intersection of outer fold ids will be used",
                        type=int, nargs='+')
    parser.add_argument('--inner-fold-idxs',
                        help="Which inner fold indices to compare. If none is give, "
                             "the intersection of outer fold indices will be used",
                        type=int, nargs='+')
    parser.add_argument('--per-site')
    parser.add_argument('--output-dir', type=Path, default=Path())
    args = parser.parse_args()

    performances = [pd.read_csv(performance_file) for performance_file in args.performance_files]
    # Determine the fold indices to look at
    site_performances = defaultdict(lambda: defaultdict(dict))
    for performance in performances:
        for index,row in performance.iterrows():
            site_id = row['site_id']
            mean_absolute_error = row['mean_absolute_error']
            r_squared = row['r_squared']
            inner_fold_id = row['inner_fold_id']
            outer_fold_id = row['outer_fold_id']
            nwp_model = row['nwp_model']
            n_test_forecasts = row['n_test_forecasts']
            n_train_forecasts = row['n_train_forecasts']
            site_performances[site_id][outer_fold_id, inner_fold_id][nwp_model] = row
    site_diffs = defaultdict(dict)
    for site_id, outer_inner_performances in site_performances.items():
        for (outer_fold_id, inner_fold_id), model_performances in outer_inner_performances.items():
            if not len(model_performances) > 1:
                #print(f"Site {site_id}, outer fold {outer_fold_id}, inner fold {inner_fold_id} only has one model: {model_performances.keys()}")
                continue
            for model_a, model_b in itertools.combinations(sorted(model_performances.keys()), 2):
                perf_a = model_performances[model_a]['mean_absolute_error']
                perf_b = model_performances[model_b]['mean_absolute_error']
                site_diffs[f'{model_a} - {model_b}'][site_id] = perf_a - perf_b

    for model_pair, diffs in site_diffs.items():
        diffs = diffs.values()
        plt.hist(diffs, bins=20)
        plt.title(model_pair)
        plt.show()

if __name__ == '__main__':
    main()
