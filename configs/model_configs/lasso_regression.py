from sklearn.linear_model import Lasso
from windpower.models import SklearnWrapper, ModelConfig

alpha = 0.0032

model = SklearnWrapper
base_args = tuple()
base_kwargs = dict(model=Lasso, scaling=False, alpha=alpha, max_iter=5000)

config = ModelConfig(model=model, model_args=base_args, model_kwargs=base_kwargs)
