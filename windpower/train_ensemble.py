import json
from tqdm import tqdm
from windpower.utils import timestamp
from windpower.dataset import SiteDataset

import argparse
import csv
import json
import pickle
from collections import defaultdict, deque

from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from windpower.dataset import GFSSiteDataset

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics

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
    def __init__(self, *args, model, **kwargs):
        self.model = model(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def fit(self, batch):
        x = batch['x']
        y = batch['y']
        self.model.fit(x, y)

    def evaluate(self, batch):
        x = batch['x']
        y = batch['y']
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
        return [mse_metric, mae_metric, rmse_metric, mad_metric, r_squared_metric]

    def save(self, save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(self.model, fp)


def train(site_files, experiment_dir, window_length, target_lag, n_estimator_values,
          max_depth_values, weather_variables, include_lead_time, include_time_of_day,
          outer_folds, outer_xval_loops, inner_folds, inner_xval_loops,
          hp_search_iterations):
    experiment_dir = experiment_dir / timestamp()
    experiment_dir.mkdir(parents=True)
    metadata = dict(window_length=window_length,
                    target_lag=target_lag,
                    n_estimator_values=n_estimator_values,
                    max_depth_values=max_depth_values,
                    weather_variables=weather_variables,
                    include_lead_time=include_lead_time,
                    include_time_of_day=include_time_of_day
                    )
    with open(experiment_dir / 'metadata.json', 'w') as fp:
        json.dump(metadata, fp)

    for site_dataset_path in tqdm(site_files):
        site_dataset = SiteDataset(dataset_path=site_dataset_path,
                                   window_length=window_length,
                                   production_offset=target_lag,
                                   weather_variables=weather_variables,
                                   include_lead_time=include_lead_time,
                                   include_time_of_day=include_time_of_day,
                                   use_cache=True)
        site_id = site_dataset.get_id()
        site_dir = experiment_dir / site_id
        site_dir.mkdir()

        base_model = EnsembleModelWrapper
        base_args = tuple()
        n_estimators = DiscreteHyperParameter(n_estimator_values)
        max_depth = DiscreteHyperParameter(max_depth_values)
        metadata['site_dataset_path'] = site_dataset_path
        base_kwargs = dict(model=RandomForestRegressor, n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        train_kwargs = dict(max_epochs=1,  # The random forest regressor doesn't do epochs
                            metadata=metadata,
                            keep_snapshots=False, )

        for i, (test_dataset, train_dataset) in tqdm(enumerate(site_dataset.k_fold_split(outer_folds)),
                                                     total=outer_folds):
            if outer_xval_loops is not None and i >= outer_xval_loops:
                break
            fold_dir = site_dir / f'outer_fold_{i:02}'
            fold_dir.mkdir(parents=True)
            test_reference_times = test_dataset.get_reference_times()
            train_reference_time = train_dataset.get_reference_times()
            np.savez(fold_dir / 'fold_reference_times.npz', train=train_reference_time, test=test_reference_times)

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
                    fit_dataset = DatasetWrapper(fit_dataset[:])  # This will return the whole dataset as a numpy array
                    validation_dataset = DatasetWrapper(validation_dataset[:])
                    hp_trainer.train(n=hp_search_iterations,
                                     training_dataset=fit_dataset,
                                     evaluation_dataset=validation_dataset,
                                     output_dir=output_dir,
                                     **train_kwargs)
                best_args, best_kwargs = hp_trainer.get_best_hyper_params()
            model = base_model(*best_args, **best_kwargs)
            train_dataset = DatasetWrapper(train_dataset[:])
            test_dataset = DatasetWrapper(test_dataset[:])
            best_performance, best_model_path = mltrain.train.train(model=model,
                                                                    training_dataset=train_dataset,
                                                                    evaluation_dataset=test_dataset,
                                                                    output_dir=fold_dir,
                                                                    **train_kwargs)