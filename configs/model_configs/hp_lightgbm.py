from lightgbm import LGBMRegressor
from windpower.models import LightGBMWrapper, ModelConfig
from mltrain.hyperparameter import DiscreteHyperParameter, GeometricHyperParameter, IntegerRangeHyperParameter

n_estimators = 1000 # We're using early stopping, so this doesn't seem to matter much. Most models converge before 100 iterations
learning_rate = GeometricHyperParameter(0.01, 0.5)
num_leaves = IntegerRangeHyperParameter(140, 1024)
max_depth = -1 #IntegerRangeHyperParameter(-1, 30)
boosting_type = 'gbdt' #DiscreteHyperParameter(['gbdt', 'dart'])  # Early stopping doesn't work with dart
objective = DiscreteHyperParameter(['regression', 'regression_l1'])  # Early stopping doesn't work with dart'regression',
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
                   eval_metric=['l1'],
                   n_jobs=-1)

model_config = ModelConfig(model=model,
                           model_args=base_args,
                           model_kwargs=base_kwargs)