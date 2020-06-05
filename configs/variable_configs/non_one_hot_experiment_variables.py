import numpy as np
from windpower.dataset import Variable, CategoricalVariable, DiscretizedVariableEvenBins

variable_definitions = {
    'DWD_ICON-EU': {'T': Variable('T'),
                    'U': Variable('U'),
                    'V': Variable('V'),
                    'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                       one_hot_encode=False),
                    'r': Variable('r'),
                    'site_production': Variable('site_production'),
                    },
    "FMI_HIRLAM": {
        "Temperature": Variable("Temperature"),
        "WindUMS": Variable("WindUMS"),
        "WindVMS": Variable("WindVMS"),
        'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                           one_hot_encode=False),
        'r': Variable('r'),
        'site_production': Variable('site_production'),
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
        'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                           one_hot_encode=False),
        'r': Variable('r'),
        'site_production': Variable('site_production'),
    }
}


