import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np

from windpower.dataset import SiteDatasetMetadata

def main():
    parser = argparse.ArgumentParser(description="Summarize performance over lead time based on test predictions")
    parser.add_argument('performance_folder_a', help="Scan these directories for experiments to use for the first model in the comparison", type=Path)
    parser.add_argument('performance_folder_b', help="Scan these directories for experiments to use for the second model in the comparison", type=Path)
    parser.add_argument('--model-a-name', help="Name to assign the first model")
    parser.add_argument('--model-b-name', help="Name to assign the first model")
    parser.add_argument('--hostname-tag', help="If set, tag each filename with the hostname of the computer",
                        action='store_true')
    parser.add_argument('--output-dir', help="Write files to this directory", type=Path, default=Path())

    args = parser.parse_args()

    model_a_name = args.model_a_name
    if model_a_name is None:
        model_a_name = args.performance_folder_a.name
    model_b_name = args.model_b_name
    if model_b_name is None:
        model_b_name = args.performance_folder_b.name

    performance_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for (model_name, experiment_directory) in [(model_a_name, args.performance_folder_a), (model_b_name, args.performance_folder_b)]:

        for outer_fold_dir in tqdm(list(experiment_directory.glob('**/outer_fold_*')), desc="Outer folds"):
            best_model_dir = outer_fold_dir / 'best_model'
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
                valid_times = predictions['target_times']
                maes = np.abs(predictions['y'].flatten() - predictions['y_hat'].flatten())

                site_dataset = Path(metadata['experiment_config']['site_dataset_path'])
                site_dataset_metadata = SiteDatasetMetadata.fromstr(site_dataset.name)
                nwp_model = site_dataset_metadata.nwp_model.identifier
                site_id = site_dataset_metadata.site_id

                for (lead_time, time_of_day, valid_time, y, y_hat, mae) in zip(lead_time, time_of_day, valid_times, predictions['y'], predictions['y_hat'], maes):
                    performance_data[site_id][nwp_model][valid_time][lead_time][model_name] = dict(y=y, y_hat=y_hat, mae=mae)

    # Yeah, perhaps a pandas query would have been better...
    paired_performance_data = []
    fieldnames = set()
    for site_id, nwp_model_performance_data in performance_data.items():
        for nwp_model, valid_time_performance_data in nwp_model_performance_data.items():
            for valid_time, lead_time_performance_data in valid_time_performance_data.items():
                for lead_time, model_performance_data in lead_time_performance_data.items():
                    if len(model_performance_data) == 2:
                        model_a_performance_dicts = model_performance_data[model_a_name]
                        model_b_performance_dicts = model_performance_data[model_b_name]
                        model_a_performance = model_a_performance_dicts['mae']
                        model_b_performance = model_b_performance_dicts['mae']
                        row = dict(model_a_mae=model_a_performance, model_b_mae=model_b_performance, site_id=site_id, nwp_model=nwp_model, valid_time=valid_time, lead_time=lead_time)
                        fieldnames.update(row.keys())
                        paired_performance_data.append(row)

    performance_name = 'hourly_paired_performance_data.csv'
    if args.hostname_tag:
        import platform
        hostname = platform.node()
        performance_name = hostname + '_' + performance_name

    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir / performance_name, 'w') as out_fp:
        csv_writer = csv.DictWriter(out_fp, fieldnames=sorted(fieldnames))
        csv_writer.writeheader()
        csv_writer.writerows(paired_performance_data)


if __name__ == '__main__':
    main()