import argparse
import pickle
from pathlib import Path
import json
import re
import numpy as np
from csv import DictWriter, DictReader
from windpower.dataset import get_nwp_model_from_path, get_site_id
import mltrain.train

def main():
    parser = argparse.ArgumentParser(description="Summarize hyper parameter performance data to a csv ")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    experiments = []
    for d in args.experiment_directories:
        for outer_fold_dir in d.glob('**/outer_fold_*'):
            latest_training = (outer_fold_dir / 'latest_experiment')
            if latest_training.exists():
                experiment_data = dict()
                best_performance_path = latest_training / 'best_performance.csv'
                if best_performance_path.exists():
                    with open(best_performance_path) as in_fp:
                        best_performance = next(iter(DictReader(in_fp)))  # take the first row, it's the only one
                        for k,v in best_performance.items():
                            experiment_data[k] = v
                metadata_path = latest_training / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as fp:
                        metadata = json.load(fp)
                    experiment_data['model'] = metadata['model_metadata']['model']
                    model_kwargs = metadata['model_metadata']['kwargs']
                    for kwarg, value in model_kwargs.items():
                        experiment_data[kwarg] = value
                    site_dataset_path = Path(metadata['experiment_config']['site_dataset_path'])
                    site_id = get_site_id(site_dataset_path)
                    nwp_model = get_nwp_model_from_path(site_dataset_path)
                    experiment_data['site_id'] = site_id
                    experiment_data['nwp_model'] = nwp_model
                fold_reference_times_path = outer_fold_dir / 'fold_reference_times.npz'
                if fold_reference_times_path.exists():
                    fold_reference_times = np.load(fold_reference_times_path)
                    training_reference_times = fold_reference_times['train']
                    test_reference_times = fold_reference_times['test']
                    experiment_data['n_train_forecasts'] = len(training_reference_times)
                    experiment_data['n_test_forecasts'] = len(test_reference_times)
                m = re.match(r'outer_fold_(\d+)', outer_fold_dir.name)
                if m is not None:
                    fold_id, = m.groups()
                    experiment_data['fold_id'] = int(fold_id)

                experiments.append(experiment_data)

    for d in args.experiment_directories:
        for outer_fold_dir in d.glob('**/outer_fold_*'):
            for inner_fold_dir in outer_fold_dir.glob('inner_fold_*'):
                for experiment in inner_fold_dir.glob('20*'):  # 20* to exclude 'latest_expriment' symlink
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
                        experiment_data['model'] = metadata['model_metadata']['model']
                        model_kwargs = metadata['model_metadata']['kwargs']
                        for kwarg, value in model_kwargs.items():
                            experiment_data[kwarg] = value
                    best_model_path = experiment / 'best_model'
                    if best_model_path.exists():
                        with open(best_model_path, 'rb') as fp:
                            best_model = pickle.load(fp)
                        if isinstance(best_model, mltrain.train.BaseModel):
                            model_metadata = best_model.get_metadata()
                            for k, v in model_metadata.items():
                                if k == 'args':
                                    for i, arg in enumerate(v):
                                        experiment_data[f'args_{i}'] = arg
                                elif k == 'kwargs':
                                    for kwarg_name, kwarg in v.items():
                                        experiment_data[f'kwarg_{kwarg_name}'] = kwarg
                                else:
                                    experiment_data[k] = v
                        try:
                            best_iteration = best_model.best_iteration_
                            experiment_data['best_iteration'] = best_iteration
                        except AttributeError:
                            pass
                    best_performance_path = experiment / 'best_performance.csv'
                    if best_performance_path.exists():
                        with open(best_performance_path) as in_fp:
                            best_performance = next(
                                iter(DictReader(in_fp)))  # take the first row, it's the only one
                            for k, v in best_performance.items():
                                experiment_data[k] = v
                    else:
                        continue  # Points without performance are useless
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