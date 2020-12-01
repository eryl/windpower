import copy
import json
import pickle
import shutil
from tqdm import tqdm
import re
from dataclasses import dataclass
from windpower.utils import timestamp, load_module
from windpower.dataset import SiteDataset, k_fold_split_reference_times, get_nwp_model_from_path, get_reference_time, get_site_id, distance_split_reference_times
import windpower.models
from typing import Union, Optional, List, Tuple
from csv import DictReader

from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
np.seterr(all='warn')

import numpy as np

import mltrain.train
from mltrain.util import load_config
from mltrain.train import TrainingArguments, TrainingConfig
from mltrain.hyperparameter import HyperParameterTrainer, HyperParameterManager
from windpower.utils import load_config
from windpower.dataset import DatasetConfig, VariableConfig, SplitConfig
from windpower.models import ModelConfig

@dataclass
class HPConfig(object):
    hp_search_iterations: int = 1


@dataclass
class HPSettings(object):
    train_config: mltrain.train.TrainingConfig
    dataset_config: DatasetConfig
    variables_config: VariableConfig
    model_config: ModelConfig
    train_times: np.ndarray
    test_times: np.ndarray
    output_dir: Path


def train(*,
          site_files,
          splits_files_list,
          experiment_dir,
          config_path: Path,
          dataset_rng=None,
          metadata=None,
          hp_rng=None):

    if hp_rng is None:
        hp_rng = np.random.RandomState()
    if metadata is None:
        metadata = dict()

    cleaned_site_files = []
    for f in site_files:
        if f.is_dir():
            cleaned_site_files.extend(f.glob('**/*.nc'))
        else:
            cleaned_site_files.append(f)
    if not cleaned_site_files:
        print(f"No site files in site dataset files in {site_files}")
    site_files = cleaned_site_files

    splits_files = dict()
    for split_file in splits_files_list:
        with open(split_file, 'rb') as fp:
            split_id = pickle.load(fp)['site_id']
            splits_files[split_id] = split_file

    model_config = load_config(config_path, ModelConfig)
    ml_model = model_config.model.__name__

    dataset_config = load_config(config_path, DatasetConfig)
    variables_config = load_config(config_path, VariableConfig)
    training_config = load_config(config_path, TrainingConfig)
    hp_config = load_config(config_path, HPConfig)

    for site_dataset_path in tqdm(sorted(site_files), desc="Sites"):
        site_id = get_site_id(site_dataset_path)
        if site_id not in splits_files:
            print(f"Not training on site {site_id}, no splits file found")
            continue
        else:
            splits_file = splits_files[site_id]
            with open(splits_file, 'rb') as fp:
                splits_data = pickle.load(fp)
                splits = splits_data['splits']
                split_config = splits_data['split_config']

        nwp_model = get_nwp_model_from_path(site_dataset_path)

        site_dir = experiment_dir / ml_model / site_id / nwp_model.identifier / timestamp()
        site_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, site_dir / 'config.py')
        site_metadata = copy.deepcopy(metadata)
        site_metadata['experiment_config'] = {'config': str(config_path),
                                              'site_dataset_path': str(site_dataset_path),
                                              'split_path': str(splits_file),}

        train_on_site(site_dataset_path=site_dataset_path,
                      site_dir=site_dir,
                      dataset_config=dataset_config,
                      variables_config=variables_config,
                      model_config=model_config,
                      training_config=training_config,
                      hp_config=hp_config,
                      splits=splits,
                      split_config=split_config,
                      hp_rng=hp_rng,
                      metadata=site_metadata,)


def train_on_site(*,
                  site_dataset_path: Path,
                  site_dir,
                  dataset_config:DatasetConfig,
                  variables_config: VariableConfig,
                  model_config: ModelConfig,
                  training_config: mltrain.train.TrainingConfig,
                  hp_config: HPConfig,
                  splits,
                  split_config: SplitConfig,
                  metadata: dict,
                  hp_rng):

    def prepare_settings(settings: HPSettings):
        train_dataset = SiteDataset(dataset_path=site_dataset_path,
                                    reference_time=settings.train_times,
                                    variables_config=settings.variables_config,
                                    dataset_config=settings.dataset_config)
        validation_dataset = SiteDataset(dataset_path=site_dataset_path,
                                         reference_time=settings.test_times,
                                         variables_config=settings.variables_config,
                                         dataset_config=settings.dataset_config)
        train_metadata = copy.copy(metadata)
        train_metadata['hp_settings'] = settings

        base_model = settings.model_config.model
        model = base_model(*settings.model_config.model_args,
                           training_dataset=train_dataset,
                           validation_dataset=validation_dataset,
                           **settings.model_config.model_kwargs)
        train_dataset = train_dataset[
                        :]  # The training script assumes the dataset is an iterator over mini-batches. We but all the data in a single batch
        validation_dataset = validation_dataset[:]
        return mltrain.train.TrainingArguments(model=model,
                                               output_dir=settings.output_dir,
                                               training_dataset=train_dataset,
                                               evaluation_dataset=validation_dataset,
                                               metadata=train_metadata,
                                               artifacts={'settings': settings,
                                                          'splits': splits,
                                                          'split_config': split_config},
                                               training_config=training_config)

    for split in splits:
        if len(split) == 3:
            i, test_reference_times, train_splits = split
            if isinstance(train_splits, list):
                # We have more than one split
                outer_fold_dir = site_dir / f'outer_fold_{i:02}'
                outer_fold_dir.mkdir(parents=True)
                np.savez(outer_fold_dir / 'fold_reference_times.npz',
                         test=test_reference_times)
                best_inner_models = []
                best_inner_params = []
                for j, validation_dataset_reference_times, fit_dataset_reference_times in tqdm(train_splits,
                                                                                               desc="Inner folds"):
                    output_dir = outer_fold_dir / f'inner_fold_{j:02}'
                    output_dir.mkdir()
                    np.savez(output_dir / 'fold_reference_times.npz',
                             train=fit_dataset_reference_times,
                             valid=validation_dataset_reference_times)

                    inner_hp_settings = HPSettings(train_config=training_config,
                                                   dataset_config=dataset_config,
                                                   variables_config=variables_config,
                                                   model_config=model_config,
                                                   train_times=fit_dataset_reference_times,
                                                   test_times=validation_dataset_reference_times,
                                                   output_dir=output_dir)

                    inner_hp_manager = HyperParameterManager(inner_hp_settings, n=hp_config.hp_search_iterations, rng=hp_rng)
                    inner_hp_trainer = HyperParameterTrainer(hp_manager=inner_hp_manager,
                                                             setting_interpreter=prepare_settings)
                    inner_hp_trainer.train()

                    best_model = inner_hp_trainer.best_model()
                    best_params = inner_hp_trainer.best_hyper_params()
                    best_inner_models.append((j, best_model))
                    best_inner_params.append((j, best_params))
                    # Each of the inner folds has resulted in a model a best setting. To get an idea of the
                    # performance of these models we run them on the held-out set. This gives us an estimate of what CV
                    #
                for i, best_inner_model in best_inner_models:
                    best_inner_models_eval_dir = outer_fold_dir / f'best_inner_model_{i:02}'
                    evaluate_model(test_reference_times,
                                   best_inner_model,
                                   best_inner_models_eval_dir)
                for i, best_params in best_inner_params:
                    training_args = prepare_settings(best_params)
                    best_inner_params_dir = outer_fold_dir / f'best_inner_setting_{i:02}'
                    training_args.output_dir = best_inner_params_dir
                    mltrain.train.train(training_args=training_args)
            else:
                raise NotImplementedError("Only test/train split has not been implemented")
        else:
            # We have more than one split
            i, test_reference_times, validation_reference_times, train_reference_times = split
            outer_fold_dir = site_dir / f'outer_fold_{i:02}'
            outer_fold_dir.mkdir(parents=True)
            np.savez(outer_fold_dir / 'fold_reference_times.npz',
                     test=test_reference_times,
                     valid=validation_reference_times,
                     train=train_reference_times)

            inner_hp_settings = HPSettings(train_config=training_config,
                                           dataset_config=dataset_config,
                                           variables_config=variables_config,
                                           model_config=model_config,
                                           train_times=train_reference_times,
                                           test_times=validation_reference_times,
                                           output_dir=outer_fold_dir)

            inner_hp_manager = HyperParameterManager(inner_hp_settings, n=hp_config.hp_search_iterations, rng=hp_rng)
            inner_hp_trainer = HyperParameterTrainer(hp_manager=inner_hp_manager,
                                                     setting_interpreter=prepare_settings)
            inner_hp_trainer.train()

            best_model = inner_hp_trainer.best_model()
            best_params = inner_hp_trainer.best_hyper_params()
            evaluate_model(test_reference_times,
                           best_model,
                           outer_fold_dir / 'best_model')

            training_args = prepare_settings(best_params)
            best_params_dir = outer_fold_dir / f'best_setting'
            training_args.output_dir = best_params_dir
            mltrain.train.train(training_args=training_args)


def evaluate_model(test_reference_times,
                   best_inner_model_path: Path,
                   best_inner_models_eval_dir: Path):
    best_inner_model_dir = best_inner_model_path.parent
    shutil.copytree(best_inner_model_dir, best_inner_models_eval_dir)
    with open(best_inner_model_dir / 'artifacts' / 'settings.pkl', 'rb') as fp:
        settings = pickle.load(fp)
    with open(best_inner_model_dir / 'metadata.json') as fp:
        metadata = json.load(fp)
    dataset_path = metadata['experiment_config']['site_dataset_path']
    dataset_config = settings.dataset_config
    dataset_config.include_variable_config = True
    test_dataset = SiteDataset(dataset_path=Path(dataset_path),
                               reference_time=test_reference_times,
                               variables_config=settings.variables_config,
                               dataset_config=dataset_config)
    with open(best_inner_model_dir / 'best_model', 'rb') as fp:
        model = pickle.load(fp)

    test_predictions = []
    data = test_dataset[:]
    x = data['x']
    y = data['y']
    variable_info = {v: (start_i, end_i, str(var_type)) for v, (start_i, end_i, var_type) in data['variable_info'].items()}
    predictions = model.predict(x)
    test_predictions.append(predictions)
    with open(best_inner_models_eval_dir / 'test_variable_definitions.json', 'w') as fp:
        json.dump(variable_info, fp)
    np.savez(best_inner_models_eval_dir / 'test_predictions.npz', x=x, y=y, y_hat=predictions)



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
        experiment_data['nwp_model'] = nwp_model.identifier
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


def parse_experiment_directory(outer_fold_dir: Path) -> Tuple[List, List, List]:
    # Decide if this experiment has nested cross-validation or not
    inner_folds = list(outer_fold_dir.glob('inner_fold_*'))
    if inner_folds:
        return parse_nested_cv_experiment_directory(outer_fold_dir)

    else:
        return parse_flat_cv_experiment_directory(outer_fold_dir)



def parse_nested_cv_experiment_directory(outer_fold_dir: Path) -> Tuple[List, List, List]:
    outer_best_model = []
    outer_best_settings = []
    inner_fold_experiments = []

    m = re.match(r'outer_fold_(\d+)', outer_fold_dir.name)
    outer_fold_id = None
    if m is not None:
        fold_id, = m.groups()
        outer_fold_id = int(fold_id)

    # Check the best inner models performance on this outer fold
    for best_inner_model_dir in tqdm(list(outer_fold_dir.glob('best_inner_model_*')), desc="Best inner model"):
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
    for best_inner_setting_dir in tqdm(list(outer_fold_dir.glob('best_inner_setting_*')), desc="Best inner settings"):
        m = re.match(r'best_inner_setting_(\d+)', best_inner_setting_dir.name)
        inner_fold_id = None
        if m is not None:
            fold_id, = m.groups()
            inner_fold_id = int(fold_id)

        settings_experiment_dir = best_inner_setting_dir / 'latest_experiment'
        try:
            experiment_data = gather_experiment_data(settings_experiment_dir)
        except FileNotFoundError as e:
            print(f"Missing files for experiment {settings_experiment_dir}, error {e}")
            continue

        experiment_data['outer_fold_id'] = outer_fold_id
        experiment_data['inner_fold_id'] = inner_fold_id
        outer_best_settings.append(experiment_data)

    # Check all the inner folds
    for inner_fold_dir in tqdm(list(outer_fold_dir.glob('inner_fold_*')), desc='Inner fold'):
        m = re.match(r'inner_fold_(\d+)', inner_fold_dir.name)
        inner_fold_id = None
        if m is not None:
            fold_id, = m.groups()
            inner_fold_id = int(fold_id)

        # Only check inner folds whose names starts with 20, to select only timestamps and not other directories
        for experiment in tqdm(list(inner_fold_dir.glob('20*')), desc='Inner fold experiment'):
            try:
                experiment_data = gather_experiment_data(experiment)
            except FileNotFoundError as e:
                print(f"Missing files for experiment {experiment}, error {e}")
                continue
            experiment_data['outer_fold_id'] = outer_fold_id
            experiment_data['inner_fold_id'] = inner_fold_id

            inner_fold_experiments.append(experiment_data)

    return inner_fold_experiments, outer_best_model, outer_best_settings


def parse_flat_cv_experiment_directory(outer_fold_dir: Path) -> Tuple[List, List, List]:
    validation_experiments = []
    outer_best_model = []
    outer_best_settings = []

    m = re.match(r'outer_fold_(\d+)', outer_fold_dir.name)
    outer_fold_id = None
    if m is not None:
        fold_id, = m.groups()
        outer_fold_id = int(fold_id)

    # Only check inner folds whose names starts with 20, to select only timestamps and not other directories
    for experiment in tqdm(list(outer_fold_dir.glob('20*')), desc='Inner fold experiment'):
        try:
            experiment_data = gather_experiment_data(experiment)
        except FileNotFoundError as e:
            print(f"Missing files for experiment {experiment}, error {e}")
            continue
        experiment_data['outer_fold_id'] = outer_fold_id

        validation_experiments.append(experiment_data)

    best_model_dir = outer_fold_dir / 'best_model'
    if best_model_dir.exists():
        try:
            experiment_data = gather_experiment_data(best_model_dir)
            experiment_data['outer_fold_id'] = outer_fold_id
            outer_best_model.append(experiment_data)
        except FileNotFoundError as e:
            print(f"Missing files for experiment {best_model_dir}, error {e}")

    best_settings_dir = outer_fold_dir / 'best_setting' / 'latest_experiment'
    if best_settings_dir.exists():
        try:
            experiment_data = gather_experiment_data(best_settings_dir)
            experiment_data['outer_fold_id'] = outer_fold_id
            outer_best_settings.append(experiment_data)
        except FileNotFoundError as e:
            print(f"Missing files for experiment {best_settings_dir}, error {e}")

    return validation_experiments, outer_best_model, outer_best_settings
