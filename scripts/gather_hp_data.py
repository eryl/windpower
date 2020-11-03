import argparse
import pickle
from pathlib import Path
import json
import re
import numpy as np
from csv import DictWriter, DictReader
from windpower.dataset import get_nwp_model_from_path, get_site_id
from windpower.train_ensemble import parse_experiment_directory
import mltrain.train
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Summarize hyper parameter performance data to a csv ")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')
    parser.add_argument('--folder-tag', help="If set, tag each filename with the folder name",
                        action='store_true')
    parser.add_argument('--hostname-tag', help="If set, tag each filename with the hostname of the computer",
                        action='store_true')
    args = parser.parse_args()


    for d in args.experiment_directories:
        all_inner_fold_experiments = []  # Performance on inner folds
        all_outer_best_model = []  # Performance on outer folds with best models in inner folds
        all_outer_best_settings = []  # Performance on outer folds with retrained model on best settings from inner fold

        for outer_fold_dir in tqdm(list(d.glob('**/outer_fold_*')), desc="Outer folds"):
            inner_fold_experiments, outer_best_model, outer_best_settings = parse_experiment_directory(outer_fold_dir)
            all_inner_fold_experiments.extend(inner_fold_experiments)
            all_outer_best_model.extend(outer_best_model)
            all_outer_best_settings.extend(outer_best_settings)

        for performance_name, experiments in [('validation_set_performance.csv', all_inner_fold_experiments),
                                              ('test_set_best_validation_model.csv', all_outer_best_model),
                                              ('test_set_best_validation_setting.csv', all_outer_best_settings),]:
            fieldnames = set()
            for experiment_data in experiments:
                fieldnames.update(experiment_data.keys())

            fieldnames = list(sorted(fieldnames))

            if args.folder_tag:
                performance_name = d.name + '_' + performance_name

            if args.hostname_tag:
                import platform
                hostname = platform.node()
                performance_name = hostname + '_' + performance_name

            with open(d / performance_name, 'w') as out_fp:
                csv_writer = DictWriter(out_fp, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(experiments)

if __name__ == '__main__':
    main()