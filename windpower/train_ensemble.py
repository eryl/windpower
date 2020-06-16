import copy
import json
import pickle
import shutil
from tqdm import tqdm
from dataclasses import dataclass
from windpower.utils import timestamp, load_module
from windpower.dataset import SiteDataset, k_fold_split_reference_times, get_nwp_model_from_path, get_reference_time, get_site_id
import windpower.models
from typing import Union, Optional

from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
np.seterr(all='warn')

import numpy as np

import mltrain.train
from mltrain.train import TrainingArguments
from mltrain.hyperparameter import HyperParameterTrainer, HyperParameterManager
from windpower.utils import load_config
from windpower.dataset import DatasetConfig, VariableConfig
from windpower.models import ModelConfig

@dataclass
class TrainConfig(object):
    outer_folds: int
    outer_xval_loops: Optional[int]
    inner_folds: Optional[int]
    inner_xval_loops: Optional[int]
    train_kwargs: mltrain.train.TrainingConfig
    hp_search_iterations: int = 1
    fold_padding: int = 0


@dataclass
class HPSettings(object):
    train_config: mltrain.train.TrainingConfig
    dataset_config: DatasetConfig
    variables_config: VariableConfig
    model_config: ModelConfig
    train_times: np.ndarray
    test_times: np.ndarray
    output_dir: Path


def train(*, site_files,
          experiment_dir,
          dataset_config_path,
          variables_config_path,
          model_config_path,
          training_config_path,
          dataset_rng=None,
          hp_rng=None):
    if hp_rng is None:
        hp_rng = np.random.RandomState()

    cleaned_site_files = []
    for f in site_files:
        if f.is_dir():
            cleaned_site_files.extend(f.glob('**/*.nc'))
        else:
            cleaned_site_files.append(f)
    if not cleaned_site_files:
        print(f"No site files in site dataset files in {site_files}")
    site_files = cleaned_site_files
    model_config = windpower.models.get_model_config(model_config_path)
    ml_model = model_config_path.with_suffix('').name

    for site_dataset_path in tqdm(sorted(site_files), desc="Sites"):
        dataset_config = windpower.dataset.get_dataset_config(dataset_config_path)
        variables_config = windpower.dataset.get_variables_config(variables_config_path)

        site_id = get_site_id(site_dataset_path)
        nwp_model = get_nwp_model_from_path(site_dataset_path)
        reference_time = get_reference_time(site_dataset_path)

        site_dir = experiment_dir / ml_model / site_id / nwp_model / timestamp()
        site_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(variables_config_path, site_dir / 'variables_config.py')
        shutil.copy(model_config_path, site_dir / 'model_config.py')
        shutil.copy(dataset_config_path, site_dir / 'dataset_config.py')
        shutil.copy(training_config_path, site_dir / 'training_config.py')

        metadata = {
            'experiment_config': {
                'variables_config': str(variables_config_path),
                'model_config': str(model_config_path),
                'training_config': str(training_config_path),
                'dataset_config': str(dataset_config_path),
                'site_dataset_path': str(site_dataset_path)
            }
        }

        training_config = load_config(training_config_path, TrainConfig)
        fold_padding = training_config.fold_padding
        outer_folds = training_config.outer_folds
        outer_xval_loops = training_config.outer_xval_loops
        inner_folds = training_config.inner_folds
        inner_xval_loops = training_config.inner_xval_loops
        hp_search_iterations = training_config.hp_search_iterations

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
            return mltrain.train.TrainingArguments(model=model,
                                                   output_dir=settings.output_dir,
                                                   training_dataset=[train_dataset[:]],
                                                   evaluation_dataset=[validation_dataset[:]],
                                                   metadata=train_metadata,
                                                   artifacts={'settings.pkl': settings},
                                                   training_config=training_config.train_kwargs)

        for i, (test_reference_times, train_reference_time) in tqdm(
                enumerate(k_fold_split_reference_times(reference_time, outer_folds, fold_padding)),
                total=outer_folds, desc="Outer folds"):
            if outer_xval_loops is not None and i >= outer_xval_loops:
                break
            outer_fold_dir = site_dir / f'outer_fold_{i:02}'
            outer_fold_dir.mkdir(parents=True)
            np.savez(outer_fold_dir / 'fold_reference_times.npz', train=train_reference_time, test=test_reference_times)

            best_inner_models = []
            best_inner_params = []

            if inner_folds > 1:
                for j, (validation_dataset_reference_times, fit_dataset_reference_times) in tqdm(
                        enumerate(k_fold_split_reference_times(train_reference_time, inner_folds, fold_padding)),
                        total=inner_folds,
                        desc="Inner folds"):
                    if inner_xval_loops is not None and j >= inner_xval_loops:
                        break

                    output_dir = outer_fold_dir / f'inner_fold_{j:02}'
                    output_dir.mkdir()
                    np.savez(output_dir / 'fold_reference_times.npz', train=fit_dataset_reference_times,
                             test=validation_dataset_reference_times)

                    inner_hp_settings = HPSettings(train_config=training_config,
                                                   dataset_config=dataset_config,
                                                   variables_config=variables_config,
                                                   model_config=model_config,
                                                   train_times=fit_dataset_reference_times,
                                                   test_times=validation_dataset_reference_times,
                                                   output_dir=output_dir)
                    inner_hp_manager = HyperParameterManager(inner_hp_settings, n=hp_search_iterations, rng=hp_rng)
                    inner_hp_trainer = HyperParameterTrainer(hp_manager=inner_hp_manager,
                                                             setting_interpreter=prepare_settings)
                    inner_hp_trainer.train()

                    best_model = inner_hp_trainer.best_model()
                    best_params = inner_hp_trainer.best_hyper_params()
                    best_inner_models.append(best_model)
                    best_inner_params.append(best_params)
                # Each of the inner folds has resulted in a model a best setting. To get an idea of the
                # performance of these models we run them on the held-out set. This gives us an estimate of what CV
                #
                for i, best_inner_model in enumerate(best_inner_models):
                    best_inner_models_eval_dir = outer_fold_dir / f'best_inner_model_{i:02}'
                    evaluate_model(test_reference_times,
                                   best_inner_model,
                                   best_inner_models_eval_dir)
                for i, best_params in enumerate(best_inner_params):
                    training_args = prepare_settings(best_params)
                    best_inner_params_dir = outer_fold_dir / f'best_inner_setting_{i:02}'
                    training_args.output_dir = best_inner_params_dir
                    mltrain.train.train(training_args=training_args)
            else:
                for k in trange(hp_search_iterations, desc="Hyper parameter searches"):
                    train_hp_instance(train_reference_time, test_reference_times, outer_fold_dir)


def evaluate_model(test_reference_times,
                   best_inner_model_path: Path,
                   best_inner_models_eval_dir: Path):
    best_inner_model_dir = best_inner_model_path.parent
    shutil.copytree(best_inner_model_dir, best_inner_models_eval_dir)
    with open(best_inner_model_dir / 'settings.pkl', 'rb') as fp:
        settings = pickle.load(fp)
    with open(best_inner_model_dir / 'metadata.json') as fp:
        metadata = json.load(fp)
    dataset_path = metadata['experiment_config']['site_dataset_path']
    dataset_config = settings.dataset_config
    dataset_config.include_variable_config = True
    test_dataset = SiteDataset(dataset_path=dataset_path,
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


