import json
import shutil
from tqdm import tqdm
from windpower.utils import timestamp, load_module
from windpower.dataset import SiteDataset, read_variables_file
import windpower.models

import argparse
import csv
import json
import pickle
from collections import defaultdict, deque

from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
np.seterr(all='warn')

import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

import mltrain.train
from mltrain.train import DiscreteHyperParameter, HyperParameterTrainer, BaseModel, LowerIsBetterMetric, HigherIsBetterMetric


class DatasetWrapper(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        yield self.dataset

    def __len__(self):
        return len(self.dataset)


class EnsembleModelWrapper(BaseModel):
    def __init__(self, *args, model, scaling=False, **kwargs):
        self.model = model(*args, **kwargs)
        self.scaling = scaling
        self.args = args
        self.kwargs = kwargs

    def fit(self, batch):
        x = batch['x']
        y = batch['y']
        if self.scaling:
            self.x_mean = np.mean(x, axis=0)
            self.x_std = np.std(x, axis=0)
            x = (x - self.x_mean)/self.x_std
        self.model.fit(x, y)

    def evaluate(self, batch):
        x = batch['x']
        y = batch['y']
        if self.scaling:
            x = (x - self.x_mean) / self.x_std
        y_hats = self.model.predict(x)

        mse = sklearn.metrics.mean_squared_error(y, y_hats)
        rmse = np.sqrt(mse)
        mae = sklearn.metrics.mean_absolute_error(y, y_hats)
        mad = sklearn.metrics.median_absolute_error(y, y_hats)
        r_squared = sklearn.metrics.r2_score(y, y_hats)
        return {'mean_squared_error': mse,
                'root_mean_squared_error': rmse,
                'mean_absolute_error': mae,
                'median_absolute_deviance': mad,
                'r_squared': r_squared}

    def get_metadata(self):
        return dict(model=self.model.__class__.__name__, args=self.args, kwargs=self.kwargs)

    def evaluation_metrics(self):
        mse_metric = LowerIsBetterMetric('mean_squared_error')
        rmse_metric = LowerIsBetterMetric('root_mean_squared_error')
        mae_metric = LowerIsBetterMetric('mean_absolute_error')
        mad_metric = LowerIsBetterMetric('median_absolute_deviance')
        r_squared_metric = HigherIsBetterMetric('r_squared')
        return [mae_metric, mse_metric, rmse_metric, mad_metric, r_squared_metric]

    def save(self, save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(self.model, fp)


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

    base_model, base_args, base_kwargs = windpower.models.get_model_config(model_config_path)
    ml_model = model_config_path.with_suffix('').name

    for site_dataset_path in tqdm(sorted(site_files)):
        site_dataset = SiteDataset(dataset_path=site_dataset_path,
                                   variables_file=variables_config_path,
                                   dataset_config_file=dataset_config_path)
        site_id = site_dataset.get_id()
        nwp_model = site_dataset.get_nwp_model()
        site_dir = experiment_dir / site_id / nwp_model / ml_model / timestamp()
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

        training_config = load_module(training_config_path)
        outer_folds = training_config.outer_folds
        outer_xval_loops = training_config.outer_xval_loops
        inner_folds = training_config.inner_folds
        inner_xval_loops = training_config.inner_xval_loops
        hp_search_iterations = training_config.hp_search_iterations
        train_kwargs = training_config.train_kwargs

        for i, (test_dataset, train_dataset) in tqdm(enumerate(site_dataset.k_fold_split(outer_folds)),
                                                     total=outer_folds):
            if outer_xval_loops is not None and i >= outer_xval_loops:
                break
            fold_dir = site_dir / f'outer_fold_{i:02}'
            fold_dir.mkdir(parents=True)
            test_reference_times = test_dataset.get_reference_times()
            train_reference_time = train_dataset.get_reference_times()
            np.savez(fold_dir / 'fold_reference_times.npz', train=train_reference_time, test=test_reference_times)

            if inner_folds > 1:
                with HyperParameterTrainer(base_model=base_model,
                                           base_args=base_args,
                                           base_kwargs=base_kwargs) as hp_trainer:
                    for j, (validation_dataset, fit_dataset) in tqdm(
                            enumerate(train_dataset.k_fold_split(inner_folds)),
                            total=inner_folds):
                        if inner_xval_loops is not None and j >= inner_xval_loops:
                            break

                        output_dir = fold_dir / f'inner_fold_{j:02}'
                        output_dir.mkdir()
                        fit_reference_times = fit_dataset.get_reference_times()
                        validation_reference_times = validation_dataset.get_reference_times()
                        np.savez(output_dir / 'fold_reference_times.npz', train=fit_reference_times,
                                 test=validation_reference_times)
                        fit_dataset = DatasetWrapper(
                            fit_dataset[:])  # This will return the whole dataset as a numpy array
                        validation_dataset = DatasetWrapper(validation_dataset[:])
                        hp_trainer.train(n=hp_search_iterations,
                                         training_dataset=fit_dataset,
                                         evaluation_dataset=validation_dataset,
                                         output_dir=output_dir,
                                         **train_kwargs)

                    best_args, best_kwargs = hp_trainer.get_best_hyper_params()
                    model = base_model(*best_args, **best_kwargs)
            else:
                with HyperParameterTrainer(base_model=base_model,
                                           base_args=base_args,
                                           base_kwargs=base_kwargs) as hp_trainer:
                    args, kwargs = hp_trainer.get_any_hyper_params()
                    model = base_model(*args, **kwargs)  # No HP tuning taking place
            train_dataset = DatasetWrapper(train_dataset[:])
            test_dataset = DatasetWrapper(test_dataset[:])
            best_performance, best_model_path = mltrain.train.train(model=model,
                                                                    training_dataset=train_dataset,
                                                                    evaluation_dataset=test_dataset,
                                                                    output_dir=fold_dir,
                                                                    metadata=metadata,
                                                                    **train_kwargs)