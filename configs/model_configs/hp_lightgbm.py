from lightgbm import LGBMRegressor
from windpower.models import SklearnWrapper
from mltrain.train import DiscreteHyperParameter, GeometricHyperParameter, IntegerRangeHyperParameter

n_estimators = IntegerRangeHyperParameter(160, 240)
learning_rate = GeometricHyperParameter(0.1, 0.5)
num_leaves = IntegerRangeHyperParameter(140, 512)
max_depth = IntegerRangeHyperParameter(-1, 30)
boosting_type = DiscreteHyperParameter(['gbdt', 'dart'])

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=LGBMRegressor,
                   boosting_type=boosting_type,
                   n_estimators=n_estimators,
                   learning_rate=learning_rate,
                   num_leaves=num_leaves,
                   max_depth=max_depth,
                   n_jobs=-1)
