import numpy as np
from windpower.dataset import Variable, CategoricalVariable, DiscretizedVariableEvenBins

VARIABLE_DEFINITIONS = {
    'DWD_ICON-EU': {'T': Variable('T'),
                    'U': Variable('U'),
                    'V': Variable('V'),
                    'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                       one_hot_encode=False),
                    'r': Variable('r'),
                    'site_production': Variable('site_production'),
                    'lead_time': Variable('lead_time'),
                    'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                                       mapping={i: i for i in range(24)},
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
                                           mapping={i: i for i in range(24)},
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
                                                    mapping={i: i for i in range(24)},
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
                                           mapping={i: i for i in range(24)},
                                           one_hot_encode=False),
    }
}

WEATHER_VARIABLES = {
    'DWD_ICON-EU':  ['T', 'U', 'V', 'phi', 'r', 'lead_time', 'time_of_day'],
    "FMI_HIRLAM": ["Temperature", "WindUMS", "WindVMS", 'phi', 'r', 'lead_time', 'time_of_day'],
    "NCEP_GFS": ['WindUMS_Height', 'WindVMS_Height', 'Temperature_Height', 'phi', 'r', 'lead_time', 'time_of_day'],
    "MetNo_MEPS": ["x_wind_10m", "y_wind_10m", "x_wind_z", "y_wind_z", "air_pressure_at_sea_level",
                   "air_temperature_0m", "air_temperature_2m", "air_temperature_z",
                   'phi_10m', 'r_10m', 'phi_z', 'r_z', 'lead_time', 'time_of_day']
}

PRODUCTION_VARIABLE = {
    'DWD_ICON-EU':  'site_production',
    "FMI_HIRLAM": 'site_production',
    "NCEP_GFS":  'site_production',
    "MetNo_MEPS": 'site_production',
}


