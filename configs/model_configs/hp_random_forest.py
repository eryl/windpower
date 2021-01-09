from sklearn.ensemble import RandomForestRegressor
from windpower.models import SklearnWrapper
from windpower.mltrain.train import DiscreteHyperParameter, IntegerRangeHyperParameter

N_ESTIMATORS = [20, 50, 100, 200, 300]
MAX_DEPTH = [1, 14]

n_estimators = IntegerRangeHyperParameter(100,200)
max_depth = 5
decorrelate = DiscreteHyperParameter([False, True])
scaling = DiscreteHyperParameter([False, True])

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=RandomForestRegressor, n_estimators=n_estimators, scaling=scaling,
                   max_depth=max_depth, n_jobs=-1)
