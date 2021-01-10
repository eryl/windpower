import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import numpy as np

from windpower.dataset import SiteDatasetMetadata
from windpower.models import load_model

def main():
    parser = argparse.ArgumentParser(description="Summarize performance over lead time based on test predictions")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')

    parser.add_argument('--folder-tag', help="If set, tag each filename with the folder name",
                        action='store_true')
    parser.add_argument('--hostname-tag', help="If set, tag each filename with the hostname of the computer",
                        action='store_true')
    parser.add_argument('--output-dir', help="Write files to this directory", type=Path, default=Path())

    args = parser.parse_args()

    feature_rankings = defaultdict(list)
    for d in args.experiment_directories:

        for outer_fold_dir in tqdm(list(d.glob('**/outer_fold_*')), desc="Outer folds"):
            best_model_dir = outer_fold_dir / 'best_model'
            try:
                if best_model_dir.exists():
                    model = load_model(best_model_dir/'best_model')
                    with open(best_model_dir / 'test_variable_definitions.json') as fp:
                        variable_definitions = json.load(fp)

                    column_names = dict()  # map a feature index to its column name
                    for name, (col_start, col_end, var_type) in variable_definitions.items():
                        for i in range(col_start, col_end):
                            var_i = i - col_start
                            column_name = f'{name}_{var_i:03}'
                            column_names[i] = column_name

                    feature_importance = model.feature_importances_
                    max_use = np.max(feature_importance)
                    for i, importance in enumerate(feature_importance):
                        column_name = column_names[i]
                        feature_rankings[column_name].append(importance/max_use)
            except:
                continue

    feature_names_col = []
    rank_col = []
    for feature_name, rankings in feature_rankings.items():
        feature_names_col.extend([feature_name]*len(rankings))
        rank_col.extend(rankings)

    dataframe = pd.DataFrame(dict(feature=feature_names_col, rank=rank_col))
    performance_name = 'feature_importance.csv'
    if args.hostname_tag:
        import platform
        hostname = platform.node()
        performance_name = hostname + '_' + performance_name

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_dir / performance_name, index=False)


if __name__ == '__main__':
    main()