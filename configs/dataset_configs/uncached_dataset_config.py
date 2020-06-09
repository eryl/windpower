from windpower.dataset import DatasetConfig

dataset_config = DatasetConfig(
    window_length = 7,
    production_offset = 3,
    horizon = 30,
    include_variable_index = False,
)

