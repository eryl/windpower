import pickle
import time
from pathlib import Path

import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA

from windpower.mltrain.performance import LowerIsBetterMetric, HigherIsBetterMetric
from windpower.mltrain.train import FullbatchModel

from windpower.dataset import VariableType
from dataclasses import dataclass
from typing import Type, Sequence, Mapping
from windpower.utils import load_config

def load_model(model_path: Path, **kwargs):
    model_dir = model_path.parent
    with open(model_dir / 'artifacts' / 'settings.pkl', 'rb') as fp:
        settings = pickle.load(fp)

    # Loading the model is a bit more complicated than just using pickle
    model_config = settings.model_config
    model = model_config.model(*model_config.model_args, **model_config.model_kwargs, **kwargs)
    model = model.load(model_path)
    return model


@dataclass
class ModelConfig(object):
    model: Type
    model_args: Sequence
    model_kwargs: Mapping

class SklearnWrapper(FullbatchModel):
    def __init__(self, *args, model, clip_predictions=True, scaling=False, decorrelate=False,
                 training_dataset=None, validation_dataset=None, **kwargs):
        self.model = model(*args, **kwargs)
        self.scaling = scaling
        self.clip_predictions = clip_predictions
        self.decorrelate = decorrelate
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        if self.decorrelate and not self.scaling:
            print("Setting scaling=True, since decorrelate=True")
            self.scaling = True
        self.args = args
        self.kwargs = kwargs

    def prepare_dataset(self, dataset):
        return dataset[:]

    def fit_normalizer(self, x, variable_info=None):
        if self.scaling:
            if variable_info is not None:
                # Only scale continuous variables
                self.var_stats = dict()
                for var_name, (start, end, var_type) in variable_info.items():
                    if var_type == VariableType.continuous:
                        self.var_stats[(start, end)] = (np.mean(x[:, start:end], axis=0, keepdims=True),
                                                        np.std(x[:, start:end], axis=0, keepdims=True))
                for (start, end), (mean, std) in self.var_stats.items():
                    x[:, start:end] = (x[:, start:end] - mean) / std
            else:
                self.x_mean = np.mean(x, axis=0)
                self.x_std = np.std(x, axis=0)
                x = (x - self.x_mean) / self.x_std
        if self.decorrelate:
            print("Fitting PCA")
            t0 = time.time()
            self.pca = PCA()
            x = self.pca.fit_transform(x)
            print(f"PCA.fit_transform took {time.time() - t0}s")
        return x

    def normalize_data(self, x):
        if self.scaling:
            if hasattr(self, 'var_stats'):
                for (start, end), (mean, std) in self.var_stats.items():
                    x[:, start:end] = (x[:, start:end] - mean) / std
            else:
                x = (x - self.x_mean) / self.x_std
        if self.decorrelate:
            x = self.pca.transform(x)
        return x

    def fit_dataset(self, dataset):
        x = dataset['x']
        y = dataset['y']
        variable_info = dataset.get('variable_info', None)
        x = self.fit_normalizer(x, variable_info)
        self.model.fit(x, y)

    def evaluate_dataset(self, dataset):
        x = dataset['x']
        y = dataset['y']
        x = self.normalize_data(x)
        y_hats = self.model.predict(x)
        if self.clip_predictions:
            np.clip(y_hats, 0, 1)

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
        return dict(model=self.model.__class__.__name__,
                    scaling=self.scaling, clip_predictions=self.clip_predictions,
                    decorrelate=self.decorrelate,
                    args=self.args, kwargs=self.kwargs)

    def evaluation_metrics(self):
        mse_metric = LowerIsBetterMetric('mean_squared_error')
        rmse_metric = LowerIsBetterMetric('root_mean_squared_error')
        mae_metric = LowerIsBetterMetric('mean_absolute_error')
        mad_metric = LowerIsBetterMetric('median_absolute_deviance')
        r_squared_metric = HigherIsBetterMetric('r_squared')
        return [mae_metric, mse_metric, rmse_metric, mad_metric, r_squared_metric]

    def save(self, save_path: Path):
        save_path = save_path.with_suffix('.pkl')
        with open(save_path, 'wb') as fp:
            pickle.dump(self.model, fp)
        return save_path

    def load(self, model_path: Path):
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
        return model


class LightGBMWrapper(SklearnWrapper):
    def __init__(self, *args, early_stopping_rounds=None, eval_metric=('l1',), **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

    def fit_dataset(self, dataset):
        x = dataset['x']
        y = dataset['y']
        variable_info = dataset.get('variable_info', None)
        x = self.fit_normalizer(x, variable_info)
        if self.validation_dataset is not None:
            validation_data = self.validation_dataset[:]
            eval_x = validation_data['x']
            eval_y = validation_data['y']
            eval_x = self.normalize_data(eval_x)
            self.model.fit(x, y, eval_set=[(eval_x, eval_y)],
                           early_stopping_rounds=self.early_stopping_rounds,
                           eval_metric=self.eval_metric)

    def get_metadata(self):
        model_metadata = dict(model=self.model.__class__.__name__,
                              scaling=self.scaling,
                              clip_predictions=self.clip_predictions,
                              decorrelate=self.decorrelate,
                              args=self.args,
                              kwargs=self.kwargs)
        try:
            best_iteration = self.model.best_iteration_
            model_metadata['best_iteration'] = best_iteration
        except sklearn.exceptions.NotFittedError:
            pass

        return model_metadata


def get_model_config(model_path):
    """
    Loads a model configuration from a python module. The module should have the attributes "model",
    "base_args" and "base_kwargs".
    :param config_path:
    :return: the triplet (model, base_args, base_kwargs)
    """
    return load_config(model_path, ModelConfig)
