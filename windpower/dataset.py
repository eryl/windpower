from collections import defaultdict
import hashlib
import xarray as xr
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
from windpower.utils import load_module

class Variable(object):
    def __init__(self, name):
        self.name = name

    def encode(self, v):
        return v

    def decode(self, v):
        return v


class CategoricalVariable(Variable):
    def __init__(self, name, levels, mapping=None, one_hot_encode=False):
        super().__init__(name)
        self.levels = levels
        self.mapping = mapping
        if mapping is not None:
            self.inv_mapping = {i:l for l,i in mapping.items()}
        else:
            self.inv_mapping = None
        self.eye = np.eye(len(levels))
        self.one_hot_encode = one_hot_encode

    def encode(self, level):
        if self.mapping is not None:
            value = self.mapping[level]
        else:
            value = level
        if self.one_hot_encode:
            return self.eye[value]
        return value

    def decode(self, value):
        if self.one_hot_encode:
            value = np.argmax(value)
        if self.inv_mapping is not None:
            return self.inv_mapping[value]




class DiscretizedVariableEvenBins(CategoricalVariable):
    def __init__(self, name, interval, bins, dtype=np.int32, **kwargs):
        self.start, self.end = interval
        self.bins = bins
        self.step_per_bin = (self.end - self.start) / bins
        self.dtype = dtype
        levels = np.arange(bins)
        super().__init__(name, levels=levels, **kwargs)

    def encode(self, value):
        discretized = (value - self.start) / self.step_per_bin
        x = np.clip(discretized.astype(self.dtype), a_min=0, a_max=self.bins-1) # This takes care of the edge case where the value is greater than the steps
        return CategoricalVariable.encode(self, x)

    def decode(self, bins):
        # find the correct edges and take the mean between them
        return (bins+0.5)*self.step_per_bin + self.start


def read_variables_file(path):
    """
    Read and return the variable definitions, weather and production variables to use. The file should be a python
    module containing the three variables VARIABLE_DEFINITIONS, WEATHER_VARIABLES and PRODUCTION_VARIABLE. These in turn
    should be dictionaries with the different NWP models as keys, and the corresponding configurations as values.
    :param path: Path to a python file containing the variables
    :return: A truplet of (variable definition dict, weather variables, production variable)
    """


def split_datetimes(datetimes, splits, padding):
    datetime_per_fold = len(datetimes) // splits
    padding_dt = np.timedelta64(padding, 'h')
    # The first and the last fold doesn't need padding removed at their ends, so we treat them differently
    fold_times = [[i * datetime_per_fold, (i + 1) * datetime_per_fold - 1] for i in range(0, splits-1)]
    fold_times.append([(splits-1)*datetime_per_fold, len(datetimes)-1])
    all_pruned = False
    while not all_pruned:
        all_pruned = True
        for i in range(0, splits - 1):
            # Check how the folds overlap and remove an equal amount of datetimes on either side, but if an uneven
            # amount can be removed, the first fold of the pair will have one more removed
            fold_a_start, fold_a_end = fold_times[i]
            fold_b_start, fold_b_end = fold_times[i+1]
            fold_a_start_datetime = datetimes[fold_a_start]
            fold_a_end_datetime = datetimes[fold_a_end]
            fold_b_start_datetime = datetimes[fold_b_start]
            fold_b_end_datetime = datetimes[fold_b_end]
            if fold_b_end_datetime - fold_a_start_datetime < padding_dt:
                raise RuntimeError("Intervals can not be separated, padding is too large (likely dataset is too small): "
                                   f"[{fold_a_start_datetime} ,{fold_a_start_datetime}], [{fold_b_start_datetime},  {fold_b_end_datetime}]")

            dt = (fold_b_start_datetime - fold_a_end_datetime).astype('timedelta64[h]')

            if dt < padding_dt:
                all_pruned = False
                # Remove one time from the fold with the most
                if fold_a_end - fold_a_start >= fold_b_end - fold_b_start:
                    fold_a_end -= 1
                    fold_times[i][1] = fold_a_end
                else:
                    fold_b_start += 1
                    fold_times[i+1][0] = fold_b_start
    fold_lengths = [end - start for start, end in fold_times]
    return fold_times

def get_nwp_model(dataset, dataset_path):
    if 'nwp_model' in dataset.attrs:
        return dataset.attrs['nwp_model']
    else:
        import re
        pattern = re.compile(r'\d+_(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).nc')
        m = re.match(pattern, dataset_path.name)
        if m is not None:
            (model,) = m.groups()
            return model

class SiteDataset(object):
    def __init__(self, *, dataset_path: Path,  variables_file, dataset_config_file,
                 dataset=None, reference_time=None):
        if dataset is None:
            dataset = xr.open_dataset(dataset_path)
        self.nwp_model = get_nwp_model(dataset, dataset_path)
        if reference_time is not None:
            dataset = dataset.sel(reference_time=reference_time)
        self.dataset_path = dataset_path
        self.dataset = dataset.squeeze()  # For the datasets we have, latitude and longitude is only a single element. This unsqueeze removes those dimensions
        #if 'latitude' in self.dataset and len(self.dataset['latitude']) == 1:
        #    self.dataset = self.dataset.isel(latitude=0)
        #if 'longitude' in self.dataset and len(self.dataset['longitude']) == 1:
        #    self.dataset = self.dataset.isel(longitude=0)
        self.reference_time = reference_time
        self.dataset_config_file = dataset_config_file
        self.read_dataset_config(dataset_config_file)
        self.variables_file = variables_file
        self.parse_variables_file(variables_file)
        self.site_id = self.dataset.attrs['site_id']
        self.setup_xref()
        if self.use_cache:
            self.setup_cache()

    def parse_variables_file(self, variables_file):
        variables_config = load_module(variables_file)
        self.weather_variables = variables_config.WEATHER_VARIABLES[self.nwp_model]
        self.variable_definitions = variables_config.VARIABLE_DEFINITIONS[self.nwp_model]
        self.production_variable = variables_config.PRODUCTION_VARIABLE[self.nwp_model]

    def read_dataset_config(self, dataset_config_file):
        dataset_config = load_module(dataset_config_file)
        self.horizon = dataset_config.horizon
        self.window_length = dataset_config.window_length
        self.production_offset = dataset_config.target_lag
        self.include_variable_index = dataset_config.include_variable_index
        self.use_cache = dataset_config.use_cache

    def get_variable_definition(self):
        return self.variable_definitions

    def get_nwp_model(self):
        return self.nwp_model

    def setup_xref(self):
        valid_time = self.dataset['valid_time']
        n_valid_times = len(valid_time)
        if self.horizon is None:
            self.horizon = n_valid_times
        elif n_valid_times < self.horizon:
            print("Horizon is longer than valid_time, reducing horizon to {}".format(n_valid_times))
            self.horizon = n_valid_times

        self.n_windows_per_forecast = (self.horizon - self.window_length) + 1
        # We need to figure out what times we can actually predict for, which will be the intersection of the time
        # intervals of weather predictions and production
        forecast_times = self.dataset['reference_time'].values
        production_times = self.dataset['production_time'].values

        left_padding = self.production_offset  # The production offset is 0-indexed, so to the number of hours left of
                                                # the prediction hour is the same as the offset
        right_padding = (self.window_length - self.production_offset) - 1  # The production offset is 0-indexed, so we
                                                                           # need to subtract one from the difference
        left_padding_dt = np.timedelta64(left_padding, 'h')
        right_padding_dt = np.timedelta64(right_padding, 'h')
        # We can actually handle production times "left_padding" hours before the first production time due to the time lag
        start_time = max(min(forecast_times), min(production_times) - left_padding_dt)

        # We can handle production time "right_padding" number of hours after the last production  time. We need a
        # full horizon worth of production data though. Forecasts are defined to be full horizon.
        end_time = min(max(forecast_times) + np.timedelta64(self.horizon, 'h'),
                       # The actual final time of a forecast is given by the issue time plus horizon
                       max(production_times) + right_padding_dt - np.timedelta64(self.horizon, 'h'))

        # Filter out the offsets which are not covered by the dataset (this could be due to the reference time of dataset
        # having been indexed or the production times not being available
        self.forecast_production_offsets = [(forecast_time_i, production_time_i+self.production_offset)
                                            for forecast_time_i, production_time_i
                                            in enumerate(self.dataset['production_index'].values)
                                            if (production_time_i >= 0 and
                                                start_time < forecast_times[forecast_time_i] < end_time and
                                                start_time < production_times[production_time_i + self.production_offset] < end_time)]

        self.n_windows = self.n_windows_per_forecast * len(self.forecast_production_offsets)

    def setup_cache(self):
        key_hash = hashlib.md5()
        # We need to identify the correct hash. We add variables which uniquely define this dataset
        key_hash.update(bytes([self.window_length, self.horizon, self.production_offset]))
        reference_time = self.dataset['reference_time'].values
        key_hash.update(reference_time.tobytes())
        with open(self.variables_file, 'rb') as fp:
            BLOCK_SIZE = 65536
            while True:
                data = fp.read(BLOCK_SIZE)
                if not data:
                    break
                key_hash.update(data)
        cache_key = key_hash.hexdigest()
        cache_file_name = self.dataset_path.with_suffix('').name + f'_cache-{cache_key}.npz'
        cache_file = self.dataset_path.with_name(cache_file_name)
        if cache_file.exists():
            cache_store = np.load(cache_file)
            self.cache = dict(x=cache_store['x'][:],
                              y=cache_store['y'][:])
        else:
            print('Building cache')
            self.cache = self._getitem_online(slice(None, None))
            np.savez(cache_file, **self.cache)
            print('Cache built')

    def get_id(self):
        return self.site_id

    def get_reference_times(self):
        return self.dataset['reference_time'].values

    def __len__(self):
        return self.n_windows

    def __getitem__(self, item):
        if self.use_cache:
            return self._getitem_cached(item)
        else:
            return self._getitem_online(item)

    def _getitem_cached(self, item):
        d = {'x': self.cache['x'][item], 'y': self.cache['y'][item]}
        return d

    def _getitem_online(self, item):
        if hasattr(item, '__index__'):
            i = item.__index__()
            # Each forecaast has self.n_windows_per_forecast number of windows. To get the right forecast, divide by the
            # this amount. To get the window inside the forecast, take the modulo
            forecast_offset_tuple_i = i // self.n_windows_per_forecast
            forecast_i, production_i = self.forecast_production_offsets[forecast_offset_tuple_i]
            window_offset_i = i % self.n_windows_per_forecast
            window_data = self.dataset.isel(reference_time=forecast_i,
                                            valid_time=slice(window_offset_i, window_offset_i+self.window_length),
                                            production_time=window_offset_i+production_i)
            x = []
            var_i = 0
            variable_index = dict()
            for v in self.weather_variables:
                var_def = self.variable_definitions[v]
                # We treat lead time separately at the moment
                if v == 'lead_time':
                    encoded_value = np.atleast_1d(var_def.encode(window_offset_i))
                elif v == 'time_of_day':
                    date = window_data['reference_time'].values + np.timedelta64(window_offset_i, 'h')
                    time_of_day = (date.astype('datetime64[h]') - date.astype('datetime64[D]')).astype(int)
                    encoded_value = np.atleast_1d(var_def.encode(time_of_day))
                else:
                    encoded_value = var_def.encode(window_data[v].values).flatten()
                var_length = len(encoded_value)
                variable_index[v] = (var_i, var_i+var_length)
                var_i += var_length
                x.append(encoded_value)

            x = np.concatenate(x)
            y = self.variable_definitions[self.production_variable].encode(window_data[self.production_variable].values)
            data = dict(x=x,
                        y=y)
            if self.include_variable_index:
                data['variable_index'] = variable_index

            return data

        elif isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            if start is None:
                if stop is None:
                    stop = len(self)
                windows = [self._getitem_online(i) for i in trange(stop)]
            elif step is None:
                windows = [self._getitem_online(i) for i in trange(start, stop)]
            else:
                windows = [self._getitem_online(i) for i in trange(start, stop, step)]
            data = defaultdict(list)
            for window in windows:
                for k,v in window.items():
                    data[k].append(v)
            return {k: np.array(v) for k,v in data.items()}
        else:
            ValueError('Expected integer or slice object, got {}'.format(item))

    def isel(self, **kwargs):
        """
        Returns a new site_dataset indexed according to the indexers in **kwargs
        :param kwargs: The variables to index according to the given indexers
        :return: New dataset with the appropriate parts indexed
        """

    def k_fold_split(self, k, padding=12):
        """
        Create a generator object which will produce k-fold splits of the dataset. Since this is assumed to be a
        time-series dataset, folds will consist of consecutive windows which has at least *padding* hours of no overlap
        :param k:
        :param padding: The folds will have this amounts of hours of no overlap
        :return: An iterator over k pairs of datasets, the first is the left out fold, the second is the remainder
        """
        # We do things simply, just divide the forecasts into k folds and remove enough of them in the ends so that we
        # satisfy the padding condition

        kwargs = dict(dataset_path=self.dataset_path,
                      variables_file=self.variables_file,
                      dataset_config_file=self.dataset_config_file)

        forecast_times = self.dataset['reference_time'].values
        fold_intervals = split_datetimes(forecast_times, k, padding+self.horizon)

        for i in range(len(fold_intervals)):
            fold_start, fold_end = fold_intervals[i]
            fold_times = forecast_times[fold_start:fold_end]
            remainder_folds = []
            # we should
            if i > 0:
                previous_fold_start, previous_fold_end = fold_intervals[i-1]
                remainder_folds.append(forecast_times[:previous_fold_end+1])
            if i < len(fold_intervals) - 1:
                next_fold_start, next_fold_end = fold_intervals[i+1]
                remainder_folds.append(forecast_times[next_fold_start:])

            remainder_times = np.concatenate(remainder_folds)
            fold = type(self)(dataset=self.dataset, reference_time=fold_times, **kwargs)
            remainder = type(self)(dataset=self.dataset, reference_time=remainder_times, **kwargs)
            yield fold, remainder


class MultiSiteDataset(object):
    def __init__(self, datasets, *, window_length, production_offset, horizon=None,
                 weather_variables=None, production_variable='site_production'):
        self.datasets = [SiteDataset(dataset_path=d, window_length=window_length, production_offset=production_offset,
                                     horizon=horizon, weather_variables=weather_variables,
                                     production_variable=production_variable) for d in datasets]
        self.window_length = window_length
        self.horizon = horizon
        self.production_offset = production_offset
        self.weather_variables = weather_variables
        self.production_variable = production_variable
        self.n = sum(len(d) for d in self.datasets)

    def __getitem__(self, item):
        ...

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
