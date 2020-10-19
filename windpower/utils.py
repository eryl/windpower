import datetime
import importlib.util
import re
from pathlib import Path
from typing import TypeVar, Type


def timestamp():
    """
    Generates a timestamp.
    :return:
    """
    t = datetime.datetime.now().replace(microsecond=0)
    #Since the timestamp is usually used in filenames, isoformat will be invalid in windows.
    #return t.isoformat()
    # We'll use another symbol instead of the colon in the ISO format
    # YYYY-MM-DDTHH:MM:SS -> YYYY-MM-DDTHH.MM.SS
    time_format = "%Y-%m-%dT%H.%M.%S"
    return t.strftime(time_format)



def load_module(module_path):
    spec = importlib.util.spec_from_file_location("module_from_file", module_path)
    module_from_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_from_file)
    return module_from_file


def sliding_window(x, window_length, step_length=1, axis=0):
    """Use stride tricks on the array x to generate a sliding window over the submitted axis"""
    from numpy.lib.stride_tricks import as_strided
    if step_length > 1:
        raise NotImplementedError("Not implemented for step sizes greater than 1")
    axis_shape = x.shape[axis]
    axis_stride = x.strides[axis]
    n_windows = (
                        axis_shape - window_length + 1) // step_length  # To support step sizes greater than 1 we also need to
    # slice the array so it's an even size along the chosen
    # axis. Perhaps we should force the user to supply valid arrays, step length and window sizes
    # The stride along each element in a window should be the same as the current stride of the axis
    # the stride between windows should be the same times the step size
    in_window_stride = axis_stride
    between_window_stride = axis_stride * step_length
    new_shape = x.shape[:axis] + (n_windows, window_length) + x.shape[axis + 1:]
    new_stride = x.strides[:axis] + (between_window_stride, in_window_stride) + x.strides[axis + 1:]
    x2 = as_strided(x, shape=new_shape, strides=new_stride)
    return x2


T = TypeVar('T')
def load_config(training_config_path: Path, config_type: Type[T]) -> T:
    mod = load_module(training_config_path)
    for k,v in mod.__dict__.items():
        if isinstance(v, config_type):
            return v
    raise ValueError(f"File {training_config_path} does not contain any {config_type} values")



def reconstruct_dataclass(str_rep, dc):
    pattern = r"{}\((.*)\)".format(dc.__name__)
    m = re.match(pattern, str_rep)
    if m is not None:
        args, = m.groups()
        # We need to sanitize the string, converting any <Object at 0x0000> to a string
        sanitizers = [r"<([\w.]+) object at 0x[\da-f]+>", r"<class '([\w.]+)'>"]
        replaced = r"'\g<1>'"
        sanitized = args
        for sanitizer in sanitizers:
            sanitized = re.sub(sanitizer, replaced, sanitized)
        contents = eval('dict({})'.format(sanitized))
        return contents


def parse_metadata_configs(metadata):
    dataset_config_str = metadata['dataset_config']
    variables_config_str = metadata['variables_config']
    model_config_str = metadata['model_config']

    from windpower.dataset import VariableConfig, DatasetConfig
    from windpower.models import ModelConfig
    variables_config = reconstruct_dataclass(variables_config_str, VariableConfig)
    dataset_config = reconstruct_dataclass(dataset_config_str, DatasetConfig)
    model_config = reconstruct_dataclass(model_config_str, ModelConfig)
    return dict(variables_config=variables_config, model_config=model_config, dataset_config=dataset_config)
