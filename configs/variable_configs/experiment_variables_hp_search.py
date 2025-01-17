import numpy as np
from windpower.dataset import Variable, CategoricalVariable, DiscretizedVariableEvenBins, VariableConfig
from windpower.mltrain.hyperparameter import DiscreteHyperParameter

hp_weather_variables = {
    'DWD_ICON-EU':  DiscreteHyperParameter([
        ['U', 'V'],
        ['T', 'U', 'V'],
        ['T', 'U', 'V', 'lead_time',],
        ['T', 'U', 'V', 'lead_time',],
        ['T', 'U', 'V', 'time_of_day'],
        ['T', 'U', 'V', 'lead_time', 'time_of_day'],
        ['phi', 'r'],
        ['T', 'phi', 'r'],
        ['T', 'phi', 'r', 'lead_time',],
        ['T', 'phi', 'r', 'time_of_day'],
        ['T', 'phi', 'r', 'lead_time', 'time_of_day']
    ]),

    "FMI_HIRLAM": ["Temperature", "WindUMS", "WindVMS", 'phi', 'r', 'lead_time', 'time_of_day'],
    "NCEP_GFS": DiscreteHyperParameter([
        ['WindUMS_Height', 'WindVMS_Height'],
        ['Temperature_Height', 'WindUMS_Height', 'WindVMS_Height'],
        ['Temperature_Height', 'WindUMS_Height', 'WindVMS_Height', 'lead_time',],
        ['Temperature_Height', 'WindUMS_Height', 'WindVMS_Height', 'lead_time',],
        ['Temperature_Height', 'WindUMS_Height', 'WindVMS_Height', 'time_of_day'],
        ['Temperature_Height', 'WindUMS_Height', 'WindVMS_Height', 'lead_time', 'time_of_day'],
        ['phi', 'r'],
        ['Temperature_Height', 'phi', 'r'],
        ['Temperature_Height', 'phi', 'r', 'lead_time',],
        ['Temperature_Height', 'phi', 'r', 'time_of_day'],
        ['Temperature_Height', 'phi', 'r', 'lead_time', 'time_of_day']
    ]),
    "MetNo_MEPS":
        ["x_wind_10m", "y_wind_10m", "x_wind_z", "y_wind_z", "air_pressure_at_sea_level",
         "air_temperature_0m", "air_temperature_2m", "air_temperature_z",
         'phi_10m', 'r_10m', 'phi_z', 'r_z', 'lead_time', 'time_of_day']
}


hp_var_defs = variable_definitions={
        'DWD_ICON-EU': {'T': Variable('T'),
                        'U': Variable('U'),
                        'V': Variable('V'),
                        'phi': DiscreteHyperParameter(
                            [Variable('phi'),
                             DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64, one_hot_encode=False),
                             DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64, one_hot_encode=True),
                             ]
                        ),
                        'r': Variable('r'),
                        'site_production': Variable('site_production'),
                        'lead_time': Variable('lead_time'),
                        'time_of_day': DiscreteHyperParameter([Variable('time_of_day'),
                                                               CategoricalVariable('time_of_day', levels=np.arange(24),
                                                                                   one_hot_encode=False)])
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
                     'phi': DiscreteHyperParameter(
                         [Variable('phi'),
                          DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64, one_hot_encode=False),
                          DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64, one_hot_encode=True),
                          ]
                     ),
                     'r': Variable('r'),
                     'site_production': Variable('site_production'),
                     'lead_time': Variable('lead_time'),
                     'time_of_day': DiscreteHyperParameter([Variable('time_of_day'),
                                                            CategoricalVariable('time_of_day', levels=np.arange(24),
                                                                                one_hot_encode=False)])
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
        }
}

variables_config = VariableConfig(
    production_variable={
        'DWD_ICON-EU':  'site_production',
        "FMI_HIRLAM": 'site_production',
        "NCEP_GFS":  'site_production',
        "MetNo_MEPS": 'site_production',
    },
    variable_definitions=hp_var_defs,
    weather_variables=hp_weather_variables)


