import copy
import json
import pickle
import shutil
from tqdm import tqdm
from dataclasses import dataclass
from windpower.utils import timestamp, load_module
from windpower.dataset import SiteDataset, k_fold_split_reference_times, get_nwp_model_from_path, get_reference_time, get_site_id, distance_split_reference_times
import windpower.models
from typing import Union, Optional, List

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
class TrainConfig(object):
    outer_folds: int
    inner_folds: int
    train_kwargs: mltrain.train.TrainingConfig
    outer_xval_loops: Optional[int] = None
    outer_xval_loop_idxs: Optional[List[int]] = None
    inner_xval_loops: Optional[int] = None
    inner_xval_loop_idxs: Optional[List[int]] = None
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


def train(*,
          site_files,
          splits_files_list,
          experiment_dir,
          config_path: Path,
          hp_search_iterations,
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
    ml_model = model_config.model.__class__.__name__

    dataset_config = load_config(config_path, DatasetConfig)
    variables_config = load_config(config_path, VariableConfig)
    training_config = load_config(config_path, TrainingConfig)

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

        site_dir = experiment_dir / ml_model / site_id / nwp_model / timestamp()
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
                      splits=splits,
                      split_config=split_config,
                      hp_rng=hp_rng,
                      metadata=site_metadata,
                      hp_search_iterations=hp_search_iterations)


def train_on_site(*,
                  site_dataset_path: Path,
                  site_dir,
                  dataset_config:DatasetConfig,
                  variables_config: VariableConfig,
                  model_config: ModelConfig,
                  training_config: mltrain.train.TrainingConfig,
                  splits,
                  split_config: SplitConfig,
                  metadata: dict,
                  hp_rng,
                  hp_search_iterations):

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
                                                                                                 total=inner_folds,
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

                    inner_hp_manager = HyperParameterManager(inner_hp_settings, n=hp_search_iterations, rng=hp_rng)
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

            inner_hp_manager = HyperParameterManager(inner_hp_settings, n=hp_search_iterations, rng=hp_rng)
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


