from sklearn.ensemble import RandomForestRegressor
from windpower.models import SklearnWrapper
from mltrain.train import DiscreteHyperParameter

N_ESTIMATORS = [20, 50, 100, 200, 300]
MAX_DEPTH = [None, 2, 5, 10, 14, 17, 20, 30, 50]

n_estimators = DiscreteHyperParameter(N_ESTIMATORS)
max_depth = DiscreteHyperParameter(MAX_DEPTH)

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=RandomForestRegressor, n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
