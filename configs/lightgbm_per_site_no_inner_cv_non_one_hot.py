import numpy as np
from windpower.dataset import Variable, CategoricalVariable, DiscretizedVariableEvenBins, VariableConfig, DatasetConfig, SplitConfig
from windpower.train_ensemble import HPConfig

from lightgbm import LGBMRegressor
from windpower.models import ModelConfig, LightGBMWrapper

import windpower.mltrain.train

production_horizon = 36
production_offset = 3
window_length = 7
horizon = production_horizon + (window_length-production_offset)  # Add enough hours to fit all the windows

dataset_config = DatasetConfig(
    window_length=window_length,
    production_offset=production_offset,
    horizon=horizon,
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




train_kwargs = windpower.mltrain.train.TrainingConfig(max_epochs=1, keep_snapshots=False)

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
        "ECMWF_HRES": 'site_production',
        'DWD_ECMWF_NCEP': 'site_production',
        'DWD_ICON-EU#ECMWF_EPS-CF#ECMWF_HRES#NCEP_GFS' :  'site_production'
    },
    variable_definitions={
        'DWD_ICON-EU': {'T': Variable('T'),
                        'U': Variable('U'),
                        'V': Variable('V'),
                        'phi_U_V': Variable('phi_U_V'),
                        'r_U_V': Variable('r_U_V'),
                        'site_production': Variable('site_production'),
                        'lead_time': Variable('lead_time'),
                        'time_of_day': Variable('time_of_day')

                        },
        "FMI_HIRLAM": {
            "Temperature": Variable("Temperature"),
            "WindUMS": Variable("WindUMS"),
            "WindVMS": Variable("WindVMS"),
            'site_production': Variable('site_production'),
            'lead_time': Variable('lead_time'),
            'phi': Variable('phi'),
            'r': Variable('r'),
            'time_of_day': Variable('time_of_day')

        },
        "NCEP_GFS": {'WindUMS_Height': Variable('WindUMS_Height'),
                     'WindVMS_Height': Variable('WindVMS_Height'),
                     'Temperature_Height': Variable('Temperature_Height'),
                     'PotentialTemperature_Sigma': Variable('PotentialTemperature_Sigma'),
                     'WindGust': Variable('WindGust'),
                     'phi_WindUMS_Height_WindVMS_Height': Variable('phi_WindUMS_Height_WindVMS_Height'),
                     'r_WindUMS_Height_WindVMS_Height': Variable('r_WindUMS_Height_WindVMS_Height'),
                     'site_production': Variable('site_production'),
                     'lead_time': Variable('lead_time'),
                     'time_of_day': Variable('time_of_day')
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
            'phi_z': Variable('phi_z'),
            'r_z': Variable('r_z'),
            'phi_10m': Variable('phi_10m'),
            'r_10m': Variable('r_10m'),
            'site_production': Variable('site_production'),
            'lead_time': Variable('lead_time'),
            'time_of_day': Variable('time_of_day'),
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
            'phi_u10_v10': Variable('phi_u10_v10'),
            'r_u10_v10': Variable('r_u10_v10'),
            'phi_u100_v100': Variable('phi_u100_v100'),
            'r_u100_v100': Variable('r_u100_v100'),
            'lead_time': Variable('lead_time'),
            'time_of_day': Variable('time_of_day'),
            'site_production': Variable('site_production'),
        },
    "ECMWF_HRES": {
            "u10" : Variable('u10'),
            "v10": Variable('v10'),
            "u100": Variable('u100'),
            "v100": Variable('v100'),
            #"u200",
            #"v200",
            "i10fg": Variable('i10fg'),
            "t2m": Variable('t2m'),
            'phi_u10_v10': Variable('phi_u10_v10'),
            'r_u10_v10': Variable('r_u10_v10'),
            'phi_u100_v100': Variable('phi_u100_v100'),
            'r_u100_v100': Variable('r_u100_v100'),
            'lead_time': Variable('lead_time'),
            'time_of_day': Variable('time_of_day'),
            'site_production': Variable('site_production'),
        },
        'DWD_ICON-EU#ECMWF_EPS-CF#ECMWF_HRES#NCEP_GFS' :  {
            'DWD_ICON-EU$T': Variable('DWD_ICON-EU$T'),
            'DWD_ICON-EU$U': Variable('DWD_ICON-EU$U'),
            'DWD_ICON-EU$V': Variable('DWD_ICON-EU$V'),
            'ECMWF_EPS-CF$i10fg': Variable('ECMWF_EPS-CF$i10fg'),
            'ECMWF_EPS-CF$t2m': Variable('ECMWF_EPS-CF$t2m'),
            'ECMWF_EPS-CF$u10': Variable('ECMWF_EPS-CF$u10'),
            'ECMWF_EPS-CF$u100': Variable('ECMWF_EPS-CF$u100'),
            'ECMWF_EPS-CF$v10': Variable('ECMWF_EPS-CF$v10'),
            'ECMWF_EPS-CF$v100': Variable('ECMWF_EPS-CF$v100'),
            'ECMWF_HRES$i10fg': Variable('ECMWF_HRES$i10fg'),
            'ECMWF_HRES$t2m': Variable('ECMWF_HRES$t2m'),
            'ECMWF_HRES$u10': Variable('ECMWF_HRES$u10'),
            'ECMWF_HRES$u100': Variable('ECMWF_HRES$u100'),
            'ECMWF_HRES$v10': Variable('ECMWF_HRES$v10'),
            'ECMWF_HRES$v100': Variable('ECMWF_HRES$v100'),
            'NCEP_GFS$Temperature_Height': Variable('NCEP_GFS$Temperature_Height'),
            'NCEP_GFS$WindUMS_Height': Variable('NCEP_GFS$WindUMS_Height'),
            'NCEP_GFS$WindVMS_Height': Variable('NCEP_GFS$WindVMS_Height'),
            'DWD_ICON-EU$r_U_V': Variable('DWD_ICON-EU$r_U_V'),
            'DWD_ICON-EU$phi_U_V': Variable('DWD_ICON-EU$phi_U_V'),
            'ECMWF_EPS-CF$r_u10_v10': Variable('ECMWF_EPS-CF$r_u10_v10'),
            'ECMWF_EPS-CF$phi_u10_v10': Variable('ECMWF_EPS-CF$phi_u10_v10'),
            'ECMWF_HRES$r_u10_v10': Variable('ECMWF_HRES$r_u10_v10'),
            'ECMWF_HRES$phi_u10_v10': Variable('ECMWF_HRES$phi_u10_v10'),
            'ECMWF_EPS-CF$r_u100_v100': Variable('ECMWF_EPS-CF$r_u100_v100'),
            'ECMWF_EPS-CF$phi_u100_v100': Variable('ECMWF_EPS-CF$phi_u100_v100'),
            'ECMWF_HRES$r_u100_v100': Variable('ECMWF_HRES$r_u100_v100'),
            'ECMWF_HRES$phi_u100_v100': Variable('ECMWF_HRES$phi_u100_v100'),
            'NCEP_GFS$r_WindUMS_Height_WindVMS_Height': Variable('NCEP_GFS$r_WindUMS_Height_WindVMS_Height'),
            'NCEP_GFS$phi_WindUMS_Height_WindVMS_Height': Variable('NCEP_GFS$phi_WindUMS_Height_WindVMS_Height'),
            'lead_time': Variable('lead_time'),
            'time_of_day': Variable('time_of_day'),
            'site_production': Variable('site_production'),
        },

    },
    weather_variables={
        'DWD_ICON-EU':  ['T', 'U', 'V', 'phi_U_V', 'r_U_V', 'lead_time', 'time_of_day'],
        "FMI_HIRLAM": ["Temperature", "WindUMS", "WindVMS", 'phi', 'r', 'lead_time', 'time_of_day'],
        "NCEP_GFS": ['WindUMS_Height', 'WindVMS_Height', 'Temperature_Height',   'phi_WindUMS_Height_WindVMS_Height',
                     'r_WindUMS_Height_WindVMS_Height', 'lead_time', 'time_of_day'],
        "MetNo_MEPS": ["x_wind_10m", "y_wind_10m", "x_wind_z", "y_wind_z", "air_pressure_at_sea_level",
                       "air_temperature_0m", "air_temperature_2m", "air_temperature_z",
                       'phi_10m', 'r_10m', 'phi_z', 'r_z', 'lead_time', 'time_of_day'],
        "ECMWF_EPS-CF": ["u10",
                         "v10",
                         "u100",
                         "v100",
                         #"u200",
                         #"v200",
                         "i10fg",
                         "t2m",
                         'phi_u10_v10',
                         'r_u10_v10',
                         'phi_u100_v100',
                         'r_u100_v100',
                         'lead_time',
                         'time_of_day'],
        "ECMWF_HRES": ["u10",
                       "v10",
                       "u100",
                       "v100",
                       # "u200",
                       # "v200",
                       "i10fg",
                       "t2m",
                       'phi_u10_v10',
                       'r_u10_v10',
                       'phi_u100_v100',
                       'r_u100_v100',
                       'lead_time',
                       'time_of_day'],
        'DWD_ICON-EU#ECMWF_EPS-CF#ECMWF_HRES#NCEP_GFS' :  [
            'DWD_ICON-EU$T',
            'DWD_ICON-EU$U',
            'DWD_ICON-EU$V',
            'ECMWF_EPS-CF$i10fg',
            'ECMWF_EPS-CF$t2m',
            'ECMWF_EPS-CF$u10',
            'ECMWF_EPS-CF$u100',
            'ECMWF_EPS-CF$v10',
            'ECMWF_EPS-CF$v100',
            'ECMWF_HRES$i10fg',
            'ECMWF_HRES$t2m',
            'ECMWF_HRES$u10',
            'ECMWF_HRES$u100',
            'ECMWF_HRES$v10',
            'ECMWF_HRES$v100',
            'NCEP_GFS$Temperature_Height',
            'NCEP_GFS$WindUMS_Height',
            'NCEP_GFS$WindVMS_Height',
            'DWD_ICON-EU$r_U_V',
            'DWD_ICON-EU$phi_U_V',
            'ECMWF_EPS-CF$r_u10_v10',
            'ECMWF_EPS-CF$phi_u10_v10',
            'ECMWF_HRES$r_u10_v10',
            'ECMWF_HRES$phi_u10_v10',
            'ECMWF_EPS-CF$r_u100_v100',
            'ECMWF_EPS-CF$phi_u100_v100',
            'ECMWF_HRES$r_u100_v100',
            'ECMWF_HRES$phi_u100_v100',
            'NCEP_GFS$r_WindUMS_Height_WindVMS_Height',
            'NCEP_GFS$phi_WindUMS_Height_WindVMS_Height',
            'lead_time',
            'time_of_day',
        ],
    })

