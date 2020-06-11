import argparse
from pathlib import Path
import json
import re
import numpy as np
from csv import DictWriter, DictReader
from windpower.dataset import get_nwp_model_from_path, get_site_id
from windpower.utils import parse_metadata_configs

def main():
    parser = argparse.ArgumentParser(description="Summarize hyper parameter performance data to a csv ")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    experiments = []
    for d in args.experiment_directories:
        for outer_fold_dir in d.glob('**/outer_fold_*'):
            for inner_fold_dir in outer_fold_dir.glob('inner_fold_*'):
                for experiment in inner_fold_dir.glob('20*'):
                    experiment_data = dict()
                    metadata_path = experiment / 'metadata.json'
                    if metadata_path.exists():
                        with open(metadata_path) as fp:
                            metadata = json.load(fp)
                        site_dataset_path = Path(metadata['experiment_config']['site_dataset_path'])
                        site_id = get_site_id(site_dataset_path)
                        nwp_model = get_nwp_model_from_path(site_dataset_path)
                        experiment_data['site_id'] = site_id
                        experiment_data['nwp_model'] = nwp_model
                        configs = parse_metadata_configs(metadata)
                        variables_config = configs['variables_config']
                        weather_variables = variables_config['weather_variables'][nwp_model]
                        variable_definitions = variables_config['variable_definitions'][nwp_model]
                        production_variable = variables_config['production_variable'][nwp_model]
                        experiment_data['weather_variables'] = json.dumps(weather_variables, sort_keys=True)
                        experiment_data['variable_definitions'] = json.dumps(variable_definitions, sort_keys=True)
                        experiment_data['production_variable'] = production_variable
                    best_performance_path = experiment / 'best_performance.csv'
                    if best_performance_path.exists():
                        with open(best_performance_path) as in_fp:
                            best_performance = next(
                                iter(DictReader(in_fp)))  # take the first row, it's the only one
                            for k, v in best_performance.items():
                                experiment_data[k] = v
                    fold_reference_times_path = experiment / 'fold_reference_times.npz'
                    if fold_reference_times_path.exists():
                        fold_reference_times = np.load(fold_reference_times_path)
                        training_reference_times = fold_reference_times['train']
                        test_reference_times = fold_reference_times['test']
                        experiment_data['n_train_forecasts'] = len(training_reference_times)
                        experiment_data['n_test_forecasts'] = len(test_reference_times)
                    m = re.match(r'outer_fold_(\d+)', outer_fold_dir.name)
                    if m is not None:
                        fold_id, = m.groups()
                        experiment_data['outer_fold_id'] = int(fold_id)

                    m = re.match(r'inner_fold_(\d+)', inner_fold_dir.name)
                    if m is not None:
                        fold_id, = m.groups()
                        experiment_data['inner_fold_id'] = int(fold_id)
                    experiments.append(experiment_data)

    fieldnames = set()
    for experiment_data in experiments:
        fieldnames.update(experiment_data.keys())

    fieldnames = list(sorted(fieldnames))
    with open(args.output, 'w') as out_fp:
        csv_writer = DictWriter(out_fp, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(experiments)



if __name__ == '__main__':
    main()