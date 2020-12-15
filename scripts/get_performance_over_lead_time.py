import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Summarize performance over lead time based on test predictions")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')
    parser.add_argument('--folder-tag', help="If set, tag each filename with the folder name",
                        action='store_true')
    parser.add_argument('--hostname-tag', help="If set, tag each filename with the hostname of the computer",
                        action='store_true')
    parser.add_argument('--output-dir', help="Write files to this directory", type=Path)
    parser.add_argument('--lead-time-col', type=int, default=350)
    parser.add_argument('--time-of-day-col', type=int, default=351)

    args = parser.parse_args()


    for d in args.experiment_directories:
        all_inner_fold_experiments = []  # Performance on inner folds
        all_outer_best_model = []  # Performance on outer folds with best models in inner folds
        all_outer_best_settings = []  # Performance on outer folds with retrained model on best settings from inner fold

        for outer_fold_dir in tqdm(list(d.glob('**/outer_fold_*')), desc="Outer folds"):
            best_model_dir = outer_fold_dir / 'best_model'
            if best_model_dir.exists():
                model_predictions_path = best_model_dir / 'test_predictions.npz'
                model_variables_path = best_model_dir / 'test_variable_definitions.json'
                predictions = np.load(model_variables_path)
                x = predictions['x']
                lead_time = x[:, args.lead_time_col]
                time_of_day = x[:, args.lead_time_col]
                reference_time = time_of_day - lead_time
                mae = np.abs(predictions['y'] - predictions['y_hat'])
                df = pd.DataFrame(dict(lead_time=lead_time, time_of_day=time_of_day, reference_time=reference_time, y=predictions['y'], y_hat=predictions['y_hat'], mae=mae))
                df.to_csv(best_model_dir / 'lead_time_predictions.csv', index=False)

if __name__ == '__main__':
    main()