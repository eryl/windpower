import pickle
import importlib

import numpy as np
import sklearn.metrics

from mltrain.train import BaseModel, LowerIsBetterMetric, HigherIsBetterMetric


class SklearnWrapper(BaseModel):
    def __init__(self, *args, model, clip_predictions=True, scaling=False, **kwargs):
        self.model = model(*args, **kwargs)
        self.scaling = scaling
        self.clip_predictions = clip_predictions
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
        if self.clip_prediction:
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
                    args=self.args, kwargs=self.kwargs)

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


def get_model_config(model_path):
    """
    Loads a model configuration from a python module. The module should have the attributes "model",
    "base_args" and "base_kwargs".
    :param config_path:
    :return: the triplet (model, base_args, base_kwargs)
    """
    spec = importlib.util.spec_from_file_location("variables_definiton", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.model, model_module.base_args, model_module.base_kwargs
