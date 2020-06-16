from lightgbm import LGBMRegressor
from windpower.models import LightGBMWrapper, ModelConfig
from mltrain.hyperparameter import DiscreteHyperParameter, GeometricHyperParameter, IntegerRangeHyperParameter

n_estimators = IntegerRangeHyperParameter(160, 3000)
learning_rate = GeometricHyperParameter(0.001, 0.5)
num_leaves = IntegerRangeHyperParameter(140, 1024)
max_depth = -1 #IntegerRangeHyperParameter(-1, 30)
boosting_type = 'gbdt' #DiscreteHyperParameter(['gbdt', 'dart'])

model = LightGBMWrapper
base_args = tuple()
base_kwargs = dict(model=LGBMRegressor,
                   boosting_type=boosting_type,
                   n_estimators=n_estimators,
                   learning_rate=learning_rate,
                   num_leaves=num_leaves,
                   max_depth=max_depth,
                   early_stopping_rounds=5,
                   n_jobs=-1)

model_config = ModelConfig(model=model,
                           model_args=base_args,
                           model_kwargs=base_kwargs)