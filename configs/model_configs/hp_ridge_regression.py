from sklearn.linear_model import Ridge
from windpower.models import SklearnWrapper
from windpower.mltrain.train import DiscreteHyperParameter, GeometricHyperParameter

alpha = GeometricHyperParameter(0.0001, 300)

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=Ridge, scaling=True, alpha=alpha)
