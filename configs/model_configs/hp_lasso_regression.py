from sklearn.linear_model import Lasso
from windpower.models import SklearnWrapper
from mltrain.train import DiscreteHyperParameter, GeometricHyperParameter

alpha = GeometricHyperParameter(0.0001, 300)

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=Lasso, scaling=True, alpha=alpha)
