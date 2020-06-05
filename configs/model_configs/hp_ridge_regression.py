from sklearn.linear_model import Ridge
from windpower.models import SklearnWrapper
from mltrain.train import DiscreteHyperParameter

ALPHA_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
alpha = DiscreteHyperParameter(ALPHA_VALUES)

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=Ridge, scaling=True, alpha=alpha)
