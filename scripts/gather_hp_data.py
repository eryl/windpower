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
    parser.add_argument('--output-dir', type=Path, default=Path())
    args = parser.parse_args()

    inner_fold_experiments = []   # Performance on inner folds
    outer_best_model = []   # Performance on outer folds with best models in inner folds
    outer_best_settings = []  # Performance on outer folds with retrained model on best settings from inner fold

    for d in args.experiment_directories:
        for outer_fold_dir in d.glob('**/outer_fold_*'):
            m = re.match(r'outer_fold_(\d+)', outer_fold_dir.name)
            outer_fold_id = None
            if m is not None:
                fold_id, = m.groups()
                outer_fold_id = int(fold_id)

            # Check the best inner models performance on this outer fold
            for best_inner_model_dir in outer_fold_dir.glob('best_inner_model_*'):
                m = re.match(r'best_inner_model_(\d+)', best_inner_model_dir.name)
                inner_fold_id = None
                if m is not None:
                    fold_id, = m.groups()
                    inner_fold_id = int(fold_id)
                try:
                    experiment_data = gather_experiment_data(best_inner_model_dir)
                except FileNotFoundError as e:
                    print(f"Missing files for experiment {best_inner_model_dir}, error {e}")
                    continue
                experiment_data['outer_fold_id'] = outer_fold_id
                experiment_data['inner_fold_id'] = inner_fold_id
                outer_best_model.append(experiment_data)

            # Check the best inner models performance on this outer fold
            for best_inner_setting_dir in outer_fold_dir.glob('best_inner_setting_*'):
                m = re.match(r'best_inner_setting_(\d+)', best_inner_setting_dir.name)
                inner_fold_id = None
                if m is not None:
                    fold_id, = m.groups()
                    inner_fold_id = int(fold_id)

                settings_experiment_dir = best_inner_setting_dir/ 'latest_experiment'
                try:
                    experiment_data = gather_experiment_data(settings_experiment_dir)
                except FileNotFoundError as e:
                    print(f"Missing files for experiment {settings_experiment_dir}, error {e}")
                    continue

                experiment_data['outer_fold_id'] = outer_fold_id
                experiment_data['inner_fold_id'] = inner_fold_id
                outer_best_settings.append(experiment_data)

            # Check all the inner folds
            for inner_fold_dir in outer_fold_dir.glob('inner_fold_*'):
                m = re.match(r'inner_fold_(\d+)', inner_fold_dir.name)
                inner_fold_id = None
                if m is not None:
                    fold_id, = m.groups()
                    inner_fold_id = int(fold_id)

                # Only check inner folds whose names starts with 20, to select only timestamps and not other directories
                for experiment in inner_fold_dir.glob('20*'):
                    try:
                        experiment_data = gather_experiment_data(experiment)
                    except FileNotFoundError as e:
                        print(f"Missing files for experiment {experiment}, error {e}")
                        continue
                    experiment_data['outer_fold_id'] = outer_fold_id
                    experiment_data['inner_fold_id'] = inner_fold_id

                    inner_fold_experiments.append(experiment_data)

    for performance_name, experiments in [('inner_fold_performance.csv', inner_fold_experiments),
                                               ('outer_fold_best_inner_model.csv', outer_best_model),
                                               ('outer_fold_best_inner_setting.csv', outer_best_settings),]:
        fieldnames = set()
        for experiment_data in experiments:
            fieldnames.update(experiment_data.keys())

        fieldnames = list(sorted(fieldnames))
        with open(args.output_dir / performance_name, 'w') as out_fp:
            csv_writer = DictWriter(out_fp, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(experiments)


def gather_experiment_data(experiment):
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
        try:
            experiment_data['model'] = metadata['model_metadata']['model']
            model_kwargs = metadata['model_metadata']['kwargs']
            for kwarg, value in model_kwargs.items():
                experiment_data[kwarg] = value
        except KeyError:
            pass
        try:
            model_config = metadata['hp_settings']['model_config']
            experiment_data['model'] = model_config['model']
            model_kwargs = model_config['model_kwargs']
            for kwarg, value in model_kwargs.items():
                experiment_data[f'model_kwarg_{kwarg}'] = str(value)
        except KeyError:
            pass
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
        raise FileNotFoundError("No performance data found")

    fold_reference_times_path = experiment.parent / 'fold_reference_times.npz'
    if fold_reference_times_path.exists():
        fold_reference_times = np.load(fold_reference_times_path)
        training_reference_times = fold_reference_times['train']
        test_reference_times = fold_reference_times['test']
        experiment_data['n_train_forecasts'] = len(training_reference_times)
        experiment_data['n_test_forecasts'] = len(test_reference_times)

    return experiment_data


if __name__ == '__main__':
    main()