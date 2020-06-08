from lightgbm import LGBMRegressor
from windpower.models import SklearnWrapper
from mltrain.train import DiscreteHyperParameter, GeometricHyperParameter, IntegerRangeHyperParameter

n_estimators = 200
learning_rate = 0.3
num_leaves = 240
max_depth = 5
boosting_type = 'dart'

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=LGBMRegressor,
                   boosting_type=boosting_type,
                   n_estimators=n_estimators,
                   learning_rate=learning_rate,
                   num_leaves=num_leaves,
                   max_depth=max_depth,
                   n_jobs=-1)
