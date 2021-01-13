import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import numpy as np

from windpower.dataset import SiteDatasetMetadata, get_nwp_model
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

    nwp_feature_rankings = defaultdict(list)
    for d in args.experiment_directories:

        for outer_fold_dir in tqdm(list(d.glob('**/outer_fold_*')), desc="Outer folds"):
            best_model_dir = outer_fold_dir / 'best_model'
            if best_model_dir.exists():
                with open(best_model_dir / 'metadata.json') as fp:
                    metadata = json.load(fp)
                    nwp_model = get_nwp_model(Path(metadata['experiment_config']['site_dataset_path']))

                feature_rankings = nwp_feature_rankings[nwp_model.identifier]
                model = load_model(best_model_dir/'best_model')
                with open(best_model_dir / 'test_variable_definitions.json') as fp:
                    variable_definitions = json.load(fp)

                column_indices = []
                variabel_indices = []
                column_names = []
                for name, (col_start, col_end, var_type) in variable_definitions.items():
                    for i in range(col_start, col_end):
                        var_i = i - col_start
                        column_names.append(name)
                        variabel_indices.append(var_i)
                        column_indices.append(i)

                sort_order = np.argsort(column_indices)
                column_indices = [column_indices[i] for i in sort_order]
                variabel_indices = [variabel_indices[i] for i in sort_order]
                column_names = [column_names[i] for i in sort_order]

                split_feature_importance = np.array(model.booster_.feature_importance('split'))
                gain_feature_importance = np.array(model.booster_.feature_importance('gain'))
                split_feature_importance = split_feature_importance / max(split_feature_importance)
                gain_feature_importance = gain_feature_importance / max(gain_feature_importance)

                feature_rankings.append(pd.DataFrame(dict(feature_index=column_indices, feature_subindex=variabel_indices, name=column_names, split=split_feature_importance, gain=gain_feature_importance)))

    for nwp_model, feature_rankings in nwp_feature_rankings.items():
        performance_name = f'feature_importance_{nwp_model}.csv'
        if args.hostname_tag:
            import platform
            hostname = platform.node()
            performance_name = hostname + '_' + performance_name


        args.output_dir.mkdir(parents=True, exist_ok=True)
        full_feature_rankings = pd.concat(feature_rankings)
        full_feature_rankings.to_csv(args.output_dir / performance_name, index=False)



if __name__ == '__main__':
    main()