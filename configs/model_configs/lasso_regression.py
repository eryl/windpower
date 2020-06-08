from sklearn.linear_model import Lasso
from windpower.models import SklearnWrapper

alpha = 0.0032

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=Lasso, scaling=True, alpha=alpha)
