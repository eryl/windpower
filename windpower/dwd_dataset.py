from collections import defaultdict
import hashlib
import xarray as xr
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
from windpower.dataset import SiteDataset, Variable, DiscretizedVariableEvenBins, CategoricalVariable


class DWDSiteDataset(SiteDataset):
    def __init__(self, *, weather_variables=('T', 'U', 'V', 'phi', 'r'), **kwargs):
        super().__init__(weather_variables=weather_variables, **kwargs)

    def setup_variables(self, variables):
        default_map = {'T': Variable('T'),
                       'U': Variable('U'),
                       'V': Variable('V'),
                       'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                          one_hot_encode=self.one_hot_encode),
                       'r': Variable('r'),
                       'site_production': Variable('site_production'),
                       'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                                          mapping={i: i for i in range(24)},
                                                          one_hot_encode=self.one_hot_encode),
                       'lead_time': Variable('lead_time')}
        if isinstance(variables, str):
            try:
                return default_map[variables]
            except KeyError:
                return super().setup_variables(variables)
        try:
            return {v: self.setup_variables(v) for v in variables}
        except TypeError:
            ## On a type error the assumption is that variables is not iterable, so a single item
            try:
                return default_map[variables]
            except KeyError:
                return super().setup_variables(variables)

class MultiSiteDataset(object):
    def __init__(self, datasets, *, window_length, production_offset, horizon=None,
                 weather_variables=('T', 'U', 'V', 'phi', 'r'), production_variable='site_production'):
        self.datasets = [SiteDataset(dataset_path=d, window_length=window_length, production_offset=production_offset,
                                     horizon=horizon, weather_variables=weather_variables,
                                     production_variable=production_variable) for d in datasets]
        self.window_length = window_length
        self.horizon = horizon
        self.production_offset = production_offset
        self.weather_variables = weather_variables
        self.production_variable = production_variable
        self.n = sum(len(d) for d in self.datasets)


def main():
    import argparse
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('site_datafile', type=Path)
    parser.add_argument('cache_file', type=Path)
    args = parser.parse_args()
    site_dataset = SiteDataset(dataset_path=args.site_datafile, window_length=7,
                               production_offset=3, use_cache=False, include_variable_index=True,
                               include_lead_time=True, include_time_of_day=True)
    for i, (fold, remainder) in enumerate(site_dataset.k_fold_split(10)):
        for j, (inner_fold, inner_remainder) in enumerate(remainder.k_fold_split(10)):
            for b in inner_fold:
                x = b['x']
                variable_index = b['variable_index']
                time_of_day_index = variable_index['time_of_day']
                time_of_day_slice = slice(*time_of_day_index)
                print('time of day', x[time_of_day_slice])
                lead_time_index = variable_index['lead_time']
                lead_time_slice = slice(*lead_time_index)
                print('lead_time:', x[lead_time_slice])
    n = len(site_dataset)

    t0 = time.time()
    data = site_dataset[0]
    x_tot = data['x']
    y_tot = data['y']

    for i in trange(1, n):
        data = site_dataset[i]
        xs = data['x']
        y = data['y']
        x_tot += xs
        y_tot += y

    x_mean = x_tot/n
    y_mean = y_tot/n

    dt = time.time() - t0
    print("Means: {}, {}. Time: {}".format(x_mean, y_mean, dt))

    array_site_dataset = np.load('/tmp/windows.npz')
    x = array_site_dataset['x']
    y = array_site_dataset['y']
    n = len(x)

    t0 = time.time()
    x_tot = x[0]
    y_tot = y[0]

    for i in trange(1, n):
        x_tot += x[i]
        y_tot += y[i]

    x_mean = x_tot / n
    y_mean = y_tot / n
    dt = time.time() - t0
    print("Means: {}, {}. Time: {}".format(x_mean, y_mean, dt))


if __name__ == '__main__':
    main()
