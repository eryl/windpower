import pickle
import time
from pathlib import Path

import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA

from mltrain.performance import LowerIsBetterMetric, HigherIsBetterMetric
from mltrain.train import BaseModel

from windpower.dataset import VariableType
from dataclasses import dataclass
from typing import Type, Sequence, Mapping
from windpower.utils import load_config

@dataclass
class ModelConfig(object):
    model: Type
    model_args: Sequence
    model_kwargs: Mapping

class SklearnWrapper(BaseModel):
    def __init__(self, *args, model, clip_predictions=True, scaling=False, decorrelate=False, **kwargs):
        self.model = model(*args, **kwargs)
        self.scaling = scaling
        self.clip_predictions = clip_predictions
        self.decorrelate = decorrelate
        if self.decorrelate and not self.scaling:
            print("Setting scaling=True, since decorrelate=True")
            self.scaling = True
        self.args = args
        self.kwargs = kwargs

    def fit(self, batch):
        x = batch['x']
        y = batch['y']
        if self.scaling:
            if 'variable_info' in batch:
                # Only scale continuous variables
                self.var_stats = dict()
                for var_name, (start, end, var_type) in batch['variable_info'].items():
                    if var_type == VariableType.continuous:
                        self.var_stats[(start, end)] = (np.mean(x[:, start:end], axis=0, keepdims=True),
                                                        np.std(x[:, start:end], axis=0, keepdims=True))
                for (start, end), (mean, std) in self.var_stats.items():
                    x[:, start:end] = (x[:, start:end]-mean)/std
            else:
                self.x_mean = np.mean(x, axis=0)
                self.x_std = np.std(x, axis=0)
                x = (x - self.x_mean)/self.x_std
        if self.decorrelate:
            print("Fitting PCA")
            t0 = time.time()
            self.pca = PCA()
            x = self.pca.fit_transform(x)
            print(f"PCA.fit_transform took {time.time() - t0}s")
        self.model.fit(x, y)

    def evaluate(self, batch):
        x = batch['x']
        y = batch['y']
        if self.scaling:
            if hasattr(self, 'var_stats'):
                for (start, end), (mean, std) in self.var_stats.items():
                    x[:, start:end] = (x[:, start:end]-mean)/std
            else:
                x = (x - self.x_mean) / self.x_std
        if self.decorrelate:
            x = self.pca.transform(x)
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
        with open(save_path.with_suffix('.pkl'), 'wb') as fp:
            pickle.dump(self.model, fp)


def get_model_config(model_path):
    """
    Loads a model configuration from a python module. The module should have the attributes "model",
    "base_args" and "base_kwargs".
    :param config_path:
    :return: the triplet (model, base_args, base_kwargs)
    """
    return load_config(model_path, ModelConfig)
