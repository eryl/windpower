import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from windpower.dataset import SiteDatasetMetadata


def main():
    parser = argparse.ArgumentParser(description="Summarize performance over lead time based on test predictions")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')

    parser.add_argument('--folder-tag', help="If set, tag each filename with the folder name",
                        action='store_true')
    parser.add_argument('--hostname-tag', help="If set, tag each filename with the hostname of the computer",
                        action='store_true')
    parser.add_argument('--output-dir', help="Write files to this directory", type=Path)

    args = parser.parse_args()


    for d in args.experiment_directories:
        lead_time_vs_performance = defaultdict(lambda: defaultdict(list))
        model_performances = []

        for outer_fold_dir in tqdm(list(d.glob('**/outer_fold_*')), desc="Outer folds"):
            best_model_dir = outer_fold_dir / 'best_model'
            try:
                if best_model_dir.exists():
                    with open(best_model_dir / 'metadata.json') as fp:
                        metadata = json.load(fp)

                    with open(best_model_dir / 'artifacts' / 'settings.pkl', 'rb') as fp:
                        settings = pickle.load(fp)

                    production_offset = settings.dataset_config.production_offset
                    model_predictions_path = best_model_dir / 'test_predictions.npz'
                    model_variables_path = best_model_dir / 'test_variable_definitions.json'
                    predictions = np.load(model_predictions_path)
                    x = predictions['x']

                    with open(model_variables_path) as fp:
                        model_variables = json.load(fp)
                    lead_time_column = model_variables['lead_time'][0]
                    time_of_day_column = model_variables['time_of_day'][0]
                    if time_of_day_column > x.shape[1]:
                        print("Warning, time of day column is higher than number of columns")
                        time_of_day_column = lead_time_column + 1

                    lead_time = x[:, lead_time_column] + production_offset
                    time_of_day = x[:, time_of_day_column]
                    reference_time = time_of_day - lead_time
                    mae = np.abs(predictions['y'] - predictions['y_hat'])

                    site_dataset = metadata['experiment_config']['site_dataset_path']
                    site_dataset_metadata = SiteDatasetMetadata.fromstr(site_dataset)
                    nwp_model = site_dataset_metadata.nwp_model.identifier

                    for (lead_time, time_of_day, reference_time, y, y_hat, mae) in zip(lead_time, time_of_day, reference_time, predictions['y'], predictions['y_hat'], mae):
                        model_performances.append(dict(nwp_model=nwp_model, lead_time=lead_time, time_of_day=time_of_day, reference_time=reference_time, y=y, y_hat=y_hat, mae=mae))

                    #df.to_csv(best_model_dir / 'lead_time_predictions.csv', index=False)
            except:
                continue

        performance_name = 'lead_time_performance.csv'
        fieldnames = ['nwp_model', 'lead_time', 'time_of_day', 'reference_time', 'y', 'y_hat', 'mae']

        if args.folder_tag:
            performance_name = d.name + '_' + performance_name

        if args.hostname_tag:
            import platform
            hostname = platform.node()
            performance_name = hostname + '_' + performance_name

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = d

        with open(output_dir / performance_name, 'w') as out_fp:
            csv_writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(model_performances)



if __name__ == '__main__':
    main()