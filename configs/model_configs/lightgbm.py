from lightgbm import LGBMRegressor
from windpower.models import ModelConfig, LightGBMWrapper
from mltrain.hyperparameter import DiscreteHyperParameter, GeometricHyperParameter, IntegerRangeHyperParameter

n_estimators = 1000  # We're using early stopping, so this doesn't seem to matter much
learning_rate = 0.15
num_leaves = 300
max_depth = -1  #IntegerRangeHyperParameter(-1, 30)
boosting_type = 'gbdt' #DiscreteHyperParameter(['gbdt', 'dart'])  # Early stopping doesn't work with dart
objective = 'regression_l1'
eval_metric = 'l1',

model = LightGBMWrapper
base_args = tuple()
base_kwargs = dict(model=LGBMRegressor,
                   objective=objective,
                   boosting_type=boosting_type,
                   n_estimators=n_estimators,
                   learning_rate=learning_rate,
                   num_leaves=num_leaves,
                   max_depth=max_depth,
                   early_stopping_rounds=5,
                   eval_metric=eval_metric,
                   n_jobs=-1)

model_config = ModelConfig(model=model,
                           model_args=base_args,
                           model_kwargs=base_kwargs)
