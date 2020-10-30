import numpy as np
from windpower.dataset import Variable, CategoricalVariable, DiscretizedVariableEvenBins, VariableConfig, DatasetConfig, SplitConfig
from windpower.train_ensemble import HPConfig

from lightgbm import LGBMRegressor
from windpower.models import ModelConfig, LightGBMWrapper

import mltrain.train

dataset_config = DatasetConfig(
    window_length=7,
    production_offset=3,
    horizon=30,
    include_variable_info=True,
)


n_estimators = 1000  # We're using early stopping, so this doesn't seem to matter much
learning_rate = 0.15
num_leaves = 300
max_depth = -1  #IntegerRangeHyperParameter(-1, 30)
boosting_type = 'gbdt' #DiscreteHyperParameter(['gbdt', 'dart'])  # Early stopping doesn't work with dart
objective = 'regression_l1'
eval_metric = 'l1',

model = LightGBMWrapper
base_args = tuple()
base_kwargs = dict(model=LGBMRegressor,
                   objective=objective,
                   boosting_type=boosting_type,
                   n_estimators=n_estimators,
                   learning_rate=learning_rate,
                   num_leaves=num_leaves,
                   max_depth=max_depth,
                   early_stopping_rounds=5,
                   eval_metric=eval_metric,
                   n_jobs=-1)

model_config = ModelConfig(model=model,
                           model_args=base_args,
                           model_kwargs=base_kwargs)




train_kwargs = mltrain.train.TrainingConfig(max_epochs=1, keep_snapshots=False)

outer_folds = 10
inner_folds = 1
outer_fold_idx = None
inner_fold_idx = None
hp_search_iterations = 1
fold_padding = 54
validation_ratio = 0.1

split_config = SplitConfig(outer_folds=outer_folds,
                           inner_folds=inner_folds,
                           outer_fold_idxs=outer_fold_idx,
                           inner_fold_idxs=inner_fold_idx,
                           validation_ratio=validation_ratio,
                           split_padding=fold_padding)


hp_config = HPConfig(hp_search_iterations=1)

variables_config = VariableConfig(
    production_variable={
        'DWD_ICON-EU':  'site_production',
        "FMI_HIRLAM": 'site_production',
        "NCEP_GFS":  'site_production',
        "MetNo_MEPS": 'site_production',
        'DWD_NCEP':  'site_production',
    "ECMWF_EPS-CF": 'site_production',
    },
    variable_definitions={
        'DWD_ICON-EU': {'T': Variable('T'),
                        'U': Variable('U'),
                        'V': Variable('V'),
                        'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                           one_hot_encode=False),
                        'r': Variable('r'),
                        'site_production': Variable('site_production'),
                        'lead_time': Variable('lead_time'),
                        'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                                           one_hot_encode=False),
                        },
        "FMI_HIRLAM": {
            "Temperature": Variable("Temperature"),
            "WindUMS": Variable("WindUMS"),
            "WindVMS": Variable("WindVMS"),
            'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                               one_hot_encode=False),
            'r': Variable('r'),
            'site_production': Variable('site_production'),
            'lead_time': Variable('lead_time'),
            'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                               one_hot_encode=False),
        },
        "NCEP_GFS": {'WindUMS_Height': Variable('WindUMS_Height'),
                     'WindVMS_Height': Variable('WindVMS_Height'),
                     'Temperature_Height': Variable('Temperature_Height'),
                     'PotentialTemperature_Sigma': Variable('PotentialTemperature_Sigma'),
                     'WindGust': Variable('WindGust'),
                     'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                        one_hot_encode=False),
                     'r': Variable('r'),
                     'site_production': Variable('site_production'),
                     'lead_time': Variable('lead_time'),
                     'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                                        one_hot_encode=False),
                     },
        "MetNo_MEPS": {
            "x_wind_10m": Variable("x_wind_10m"),
            "y_wind_10m": Variable("y_wind_10m"),
            "x_wind_z": Variable("x_wind_z"),
            "y_wind_z": Variable("y_wind_z"),
            "air_pressure_at_sea_level": Variable("air_pressure_at_sea_level"),
            "air_temperature_0m": Variable("air_temperature_0m"),
            "air_temperature_2m": Variable("air_temperature_2m"),
            "air_temperature_z": Variable("air_temperature_z"),
            'phi_z': DiscretizedVariableEvenBins('phi_z', (-np.pi, np.pi), 64,
                                                 one_hot_encode=False),
            'r_z': Variable('r_z'),
            'phi_10m': DiscretizedVariableEvenBins('phi_m10', (-np.pi, np.pi), 64,
                                                   one_hot_encode=False),
            'r_10m': Variable('r_10m'),
            'site_production': Variable('site_production'),
            'lead_time': Variable('lead_time'),
            'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                               one_hot_encode=False),
        },
        'DWD_NCEP': {
            'T': Variable('T'),
            'U': Variable('U'),
            'V': Variable('V'),
            'dwd_phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                               one_hot_encode=False),
            'dwd_r': Variable('r'),
            'WindUMS_Height': Variable('WindUMS_Height'),
            'WindVMS_Height': Variable('WindVMS_Height'),
            'Temperature_Height': Variable('Temperature_Height'),
            'PotentialTemperature_Sigma': Variable('PotentialTemperature_Sigma'),
            'WindGust': Variable('WindGust'),
            'ncep_phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                               one_hot_encode=False),
            'ncep_r': Variable('r'),
            'site_production': Variable('site_production'),
            'lead_time': Variable('lead_time'),
            'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                               one_hot_encode=False),
        },
        "ECMWF_EPS-CF": {
            "u10" : Variable('u10'),
            "v10": Variable('v10'),
            "u100": Variable('u100'),
            "v100": Variable('v100'),
            #"u200",
            #"v200",
            "i10fg": Variable('i10fg'),
            "t2m": Variable('t2m'),
            'phi_10': DiscretizedVariableEvenBins('phi_10', (-np.pi, np.pi), 64,
                                                  one_hot_encode=True),
            'r_10': Variable('r_10'),
            'phi_100': DiscretizedVariableEvenBins('phi_10', (-np.pi, np.pi), 64,
                                                   one_hot_encode=True),
            'r_100': Variable('r_10'),
            'lead_time': Variable('lead_time'),
            'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                               mapping={i: i for i in range(24)},
                                               one_hot_encode=True),
            'site_production': Variable('site_production'),
        },
    },
    weather_variables={
        'DWD_ICON-EU':  ['T', 'U', 'V', 'phi', 'r', 'lead_time', 'time_of_day'],
        "FMI_HIRLAM": ["Temperature", "WindUMS", "WindVMS", 'phi', 'r', 'lead_time', 'time_of_day'],
        "NCEP_GFS": ['WindUMS_Height', 'WindVMS_Height', 'Temperature_Height', 'phi', 'r', 'lead_time', 'time_of_day'],
        "MetNo_MEPS": ["x_wind_10m", "y_wind_10m", "x_wind_z", "y_wind_z", "air_pressure_at_sea_level",
                       "air_temperature_0m", "air_temperature_2m", "air_temperature_z",
                       'phi_10m', 'r_10m', 'phi_z', 'r_z', 'lead_time', 'time_of_day'],
        "DWD_NCEP": ['T', 'U', 'V', 'dwd_phi', 'dwd_r', 'WindUMS_Height', 'WindVMS_Height', 'Temperature_Height', 'ncep_phi', 'ncep_r', 'lead_time', 'time_of_day'],
"ECMWF_EPS-CF": ["u10",
                     "v10",
                     "u100",
                     "v100",
                     #"u200",
                     #"v200",
                     "i10fg",
                     "t2m",
                     'phi_10', 'r_10', 'phi_100', 'r_100',
                     'lead_time',
                     'time_of_day'],
    })

