import copy
import json
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
from mltrain.hyperparameter import DiscreteHyperParameter, HyperParameterTrainer, ObjectHyperParameterManager
from windpower.utils import load_config

@dataclass
class TrainConfig(object):
    outer_folds: int
    outer_xval_loops: Optional[int]
    inner_folds: Optional[int]
    inner_xval_loops: Optional[int]
    train_kwargs: mltrain.train.TrainingConfig
    hp_search_iterations: int = 1
    fold_padding: int = 0


class DatasetWrapper(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        yield self.dataset[:]

    def __len__(self):
        return len(self.dataset)


def train(*, site_files,
          experiment_dir,
          dataset_config_path,
          variables_config_path,
          model_config_path,
          training_config_path):

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

    for site_dataset_path in tqdm(sorted(site_files)):
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

        train_kwargs = training_config.train_kwargs

        training_config_hp_manager = ObjectHyperParameterManager(train_kwargs)
        model_config_hp_manager = ObjectHyperParameterManager(model_config)
        dataset_config_hp_manager = ObjectHyperParameterManager(dataset_config)
        variables_config_hp_manager = ObjectHyperParameterManager(variables_config)

        def train_hp_instance(train_times, test_times, output_dir):
            variables_config_id, variables_config_instance = variables_config_hp_manager.get_next()
            dataset_config_id, dataset_config_instance = dataset_config_hp_manager.get_next()
            training_config_id, training_config_instance = training_config_hp_manager.get_next()
            model_config_id, model_config_instance = model_config_hp_manager.get_next()
            model_base, model_args, model_kwargs = model_config_instance.model, model_config_instance.model_args, model_config_instance.model_kwargs

            model = model_base(*model_args, **model_kwargs)
            train_dataset = SiteDataset(dataset_path=site_dataset_path,
                                        reference_time=train_times,
                                        variables_config=variables_config_instance,
                                        dataset_config=dataset_config_instance)
            validation_dataset = SiteDataset(dataset_path=site_dataset_path,
                                             reference_time=test_times,
                                             variables_config=variables_config_instance,
                                             dataset_config=dataset_config_instance)
            train_metadata = copy.copy(metadata)
            train_metadata['dataset_config'] = dataset_config_instance
            train_metadata['variables_config'] = variables_config_instance
            train_metadata['model_config'] = model_config_instance
            train_metadata['training_config'] = training_config_instance

            mltrain.train.train(model=model,
                                training_dataset=[train_dataset[:]],
                                evaluation_dataset=[validation_dataset[:]],
                                training_config=training_config_instance,
                                metadata=train_metadata,
                                output_dir=output_dir)

        for i, (test_reference_times, train_reference_time) in tqdm(enumerate(k_fold_split_reference_times(reference_time,
                                                                                            outer_folds, fold_padding
                                                                                            )),
                                                     total=outer_folds):
            if outer_xval_loops is not None and i >= outer_xval_loops:
                break
            fold_dir = site_dir / f'outer_fold_{i:02}'
            fold_dir.mkdir(parents=True)
            np.savez(fold_dir / 'fold_reference_times.npz', train=train_reference_time, test=test_reference_times)

            if inner_folds > 1:
                for j, (validation_dataset_reference_times, fit_dataset_reference_times) in tqdm(
                        enumerate(k_fold_split_reference_times(train_reference_time, inner_folds, fold_padding)),
                        total=inner_folds):
                    if inner_xval_loops is not None and j >= inner_xval_loops:
                        break

                    output_dir = fold_dir / f'inner_fold_{j:02}'
                    output_dir.mkdir()
                    np.savez(output_dir / 'fold_reference_times.npz', train=fit_dataset_reference_times,
                             test=validation_dataset_reference_times)

                    for k in trange(hp_search_iterations, desc="Hyper parameter searches"):
                        train_hp_instance(fit_dataset_reference_times, validation_dataset_reference_times, output_dir)
            else:
                for k in trange(hp_search_iterations, desc="Hyper parameter searches"):
                    train_hp_instance(train_reference_time, test_reference_times, fold_dir)
