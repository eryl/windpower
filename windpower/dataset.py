from collections import defaultdict
import hashlib
import xarray as xr
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
from windpower.utils import load_module, sliding_window
from dataclasses import dataclass
from typing import List, Dict, Any

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
            if isinstance(level, np.ndarray):
                np.vectorize(self.mapping.__getitem__)(level)
            else:
                value = self.mapping[level]
        else:
            value = level
        if self.one_hot_encode:
            return self.eye[level]
        return value

    def decode(self, value):
        if self.one_hot_encode:
            value = np.argmax(value)
        if self.inv_mapping is not None:
            if isinstance(value, np.ndarray):
                np.vectorize(self.inv_mapping.__getitem__)(value)
            else:
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

@dataclass
class VariableDefinition(object):
    name: str
    type: str
    kwargs: Dict[str, Any]

@dataclass
class VariableConfig(object):
    production_variable: Dict[str, str]
    variable_definitions: Dict[str, Dict[str, Variable]]
    weather_variables: Dict[str, List[str]]

@dataclass
class DatasetConfig(object):
    horizon: int
    window_length: int
    production_offset: int
    include_variable_index: bool


def get_dataset_config(dataset_config_path):
    dataset_module = load_module(dataset_config_path)
    return dataset_module.dataset_config

def get_variables_config(variables_config_path):
    variables_module = load_module(variables_config_path)
    return variables_module.variables_config

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
        return get_nwp_model_from_path(dataset_path)


def get_nwp_model_from_path(dataset_path):
    import re
    #pattern = re.compile(r'\d+_(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).nc|.*(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).*.nc')
    pattern = re.compile(r'.*(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).*.nc')
    m = re.match(pattern, dataset_path.name)
    if m is not None:
        (model,) = m.groups()
        return model
    else:
        raise ValueError(f"Not a dataset path: {dataset_path}")


class SiteDatasetOld(object):
    def __init__(self, *,
                 dataset_path: Path,
                 dataset_config: DatasetConfig,
                 dataset=None,
                 reference_time=None):
        self.dataset_path = dataset_path
        if dataset is None:
            dataset = xr.open_dataset(dataset_path)
        self.dataset = dataset.squeeze()  # For the datasets we have, latitude and longitude is only a single element. This unsqueeze removes those dimensions
        self.nwp_model = get_nwp_model(dataset, dataset_path)
        self.reference_time = reference_time
        self.dataset_config = dataset_config
        variables_config = dataset_config.variable_config
        self.weather_variables = variables_config.weather_variables[self.nwp_model]
        self.variable_definitions = variables_config.variable_definitions[self.nwp_model]
        self.production_variable = variables_config.production_variable[self.nwp_model]
        self.horizon = dataset_config.horizon
        self.window_length = dataset_config.window_length
        self.production_offset = dataset_config.production_offset
        self.include_variable_index = dataset_config.include_variable_index
        self.use_cache = dataset_config.use_cache
        self.site_id = self.dataset.attrs['site_id']

        n_valid_times = len(self.dataset['valid_time'])
        if self.horizon is None:
            self.horizon = n_valid_times
        elif n_valid_times < self.horizon:
            print("Horizon is longer than valid_time, reducing horizon to {}".format(n_valid_times))
            self.horizon = n_valid_times
        else:
            self.dataset = self.dataset.isel(valid_time=slice(0, self.horizon))

        self.setup_reference_time(reference_time)
        #self.setup_xref()
        if self.use_cache:
            self.setup_cache()

    def parse_variables_file(self, variables_file):
        variables_config = load_module(variables_file)


    def read_dataset_config(self, dataset_config_file):
        dataset_config = load_module(dataset_config_file)

    def get_variable_definition(self):
        return self.variable_definitions

    def setup_reference_time(self, reference_time):
        if reference_time is not None:
            self.dataset = self.dataset.sel(reference_time=reference_time)

        # Filter the reference times so what all of them can be used. In particular, compare reference times to
        # production times and see which ones would be missing production data
        reference_times = self.dataset['reference_time'].values
        production_times = self.dataset['production_time'].values

        left_padding = self.production_offset  # The production offset is 0-indexed, so to the number of hours left of
        # the prediction hour is the same as the offset
        right_padding = (self.window_length - self.production_offset) - 1  # The production offset is 0-indexed, so we
        # need to subtract one from the difference
        left_padding_dt = np.timedelta64(left_padding, 'h')
        right_padding_dt = np.timedelta64(right_padding, 'h')
        # We can actually handle production times "left_padding" hours before the first production time
        # due to the time lag
        start_time = max(min(reference_times), min(production_times) - left_padding_dt)

        # We can handle production time "right_padding" number of hours after the last production  time. We need a
        # full horizon worth of production data though. Forecasts are defined to be full horizon.
        end_time = min(max(reference_times),
                       # The actual final time of a forecast is given by the issue time plus horizon
                       max(production_times) - (np.timedelta64(self.horizon, 'h') + right_padding_dt))
        selected_indices = np.logical_and(start_time <= reference_times, reference_times < end_time)
        filtered_reference_time = reference_times[selected_indices]
        self.dataset = self.dataset.sel(reference_time=filtered_reference_time)


    def get_nwp_model(self):
        return self.nwp_model

    def setup_xref(self):
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
                       max(production_times) - (np.timedelta64(self.horizon, 'h') + right_padding_dt))

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
                      dataset_config=self.dataset_config)

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


class WindowedDataset(object):
    def __init__(self, dataset: xr.Dataset, variables_config, window_config: DatasetConfig, nwp_model):
        self.nwp_model = nwp_model
        self.weather_variables = variables_config.weather_variables[self.nwp_model]
        self.variable_definitions = variables_config.variable_definitions[self.nwp_model]
        self.production_variable = variables_config.production_variable[self.nwp_model]
        self.horizon = window_config.horizon
        self.window_length = window_config.window_length
        self.production_offset = window_config.production_offset
        self.include_variable_index = window_config.include_variable_index

    def __getitem__(self, item):
        ...

class SiteDataset(object):
    def __init__(self, *,
                 dataset_path: Path,
                 dataset_config: DatasetConfig,
                 variables_config: VariableConfig,
                 dataset=None,
                 reference_time=None):
        self.dataset_path = dataset_path
        if dataset is None:
            dataset = xr.open_dataset(dataset_path)
        self.dataset = dataset.squeeze()  # For the datasets we have, latitude and longitude is only a single element. This unsqueeze removes those dimensions
        self.nwp_model = get_nwp_model(dataset, dataset_path)
        self.reference_time = reference_time
        self.variables_config = variables_config
        self.dataset_config = dataset_config
        self.weather_variables = variables_config.weather_variables[self.nwp_model]
        self.variable_definitions = variables_config.variable_definitions[self.nwp_model]
        self.production_variable = variables_config.production_variable[self.nwp_model]
        self.horizon = dataset_config.horizon
        self.window_length = dataset_config.window_length
        self.production_offset = dataset_config.production_offset
        self.include_variable_index = dataset_config.include_variable_index
        self.site_id = self.dataset.attrs['site_id']

        n_valid_times = len(self.dataset['valid_time'])
        if self.horizon is None:
            self.horizon = n_valid_times
        elif n_valid_times < self.horizon:
            print("Horizon is longer than valid_time, reducing horizon to {}".format(n_valid_times))
            self.horizon = n_valid_times
        else:
            self.dataset = self.dataset.isel(valid_time=slice(0, self.horizon))

        self.setup_reference_time(reference_time)
        self.make_memdataset()

    def get_variable_definition(self):
        return self.variable_definitions

    def setup_reference_time(self, reference_time):
        if reference_time is not None:
            self.dataset = self.dataset.sel(reference_time=reference_time)

        # Filter the reference times so what all of them can be used. In particular, compare reference times to
        # production times and see which ones would be missing production data
        reference_times = self.dataset['reference_time'].values
        production_times = self.dataset['production_time'].values

        left_padding = self.production_offset  # The production offset is 0-indexed, so to the number of hours left of
        # the prediction hour is the same as the offset
        right_padding = (self.window_length - self.production_offset) - 1  # The production offset is 0-indexed, so we
        # need to subtract one from the difference
        left_padding_dt = np.timedelta64(left_padding, 'h')
        right_padding_dt = np.timedelta64(right_padding, 'h')
        # We can actually handle production times "left_padding" hours before the first production time
        # due to the time lag
        start_time = max(min(reference_times), min(production_times) - left_padding_dt)

        # We can handle production time "right_padding" number of hours after the last production  time. We need a
        # full horizon worth of production data though. Forecasts are defined to be full horizon.
        end_time = min(max(reference_times),
                       # The actual final time of a forecast is given by the issue time plus horizon
                       max(production_times) - (np.timedelta64(self.horizon, 'h') + right_padding_dt))
        selected_indices = np.logical_and(start_time <= reference_times, reference_times < end_time)
        filtered_reference_time = reference_times[selected_indices]
        self.dataset = self.dataset.sel(reference_time=filtered_reference_time)


    def get_nwp_model(self):
        return self.nwp_model

    def get_id(self):
        return self.site_id

    def get_reference_times(self):
        return self.dataset['reference_time'].values

    def __len__(self):
        return len(self.windows)

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
                      dataset_config=self.dataset_config,
                      variables_config=self.variables_config)

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


    def make_memdataset(self):
        n_ref_times = len(self.dataset['reference_time'])
        n_valid_times = len(self.dataset['valid_time'])
        self.n_windows_per_forecast = (self.horizon - self.window_length) + 1

        var_arrays = []
        var_indices = dict()
        var_start = 0
        var_end = 0
        for var in self.weather_variables:
            if var == 'lead_time' or var == 'time_of_day':
                #Lead time and time of day are not stored in the dataset, they are derived from the reference time
                continue
            var_values = self.dataset[var].values  # Pick out the numpy array values
            if not (var_values.shape[0] == n_ref_times and var_values.shape[1] == n_valid_times):
                raise ValueError(f"Variable {var} has mismatching shape: {var_values.shape}")
            var_definition = self.variable_definitions[var]
            encoded_values = var_definition.encode(var_values)
            var_values = encoded_values.reshape(n_ref_times, n_valid_times, -1)
            var_arrays.append(var_values)
            var_length = var_values.shape[-1]
            # Since each variable is repeated window_length number of times, we set the var_index to this value here
            var_end += var_length*self.window_length
            var_indices[var] = ((var_start, var_end))
            var_start = var_end
        # We should now have an array where the first dimension is the number of forecasts, equal to the number of
        # reference times in this array. The second dimension should be valid-time, the number of hours in this forecast.
        # We're actually more interested in selecting 'horizon' hours from this valid time, and in that horizon-sized
        # forecast make sliding windows
        # So consider we have a ndarray with shape (n_reference_times, horizon, ...), we're going to create a new
        # ndarray with shape (n_reference_times, n_windows_per_forecast, window_size, ...)
        # We do this using stride tricks to fake the windows per forecast and window size
        all_var_values = np.concatenate(var_arrays, axis=-1)
        strided = sliding_window(all_var_values, window_length=self.window_length, step_length=1, axis=1)
        # Now we have an array of shape (n_reference_times, n_windows_perf_forecast, window_size, ...), we wan't to
        # collapse the window_size and trailing dimension (so each window is a single feature vector
        self.feature_vectors = strided.reshape(n_ref_times, self.n_windows_per_forecast, -1)

        # We only add the lead time of the first hour of a window (since the other are perfectly linearly dependent,
        # it would harm linear models)
        if 'lead_time' in self.weather_variables:
            var_values = np.tile(np.arange(self.n_windows_per_forecast), (n_ref_times, 1))
            var_def = self.variable_definitions['lead_time']
            encoded_values = var_def.encode(var_values)
            # We need to reshape the encoded values so that each lead time will be added at each window
            encoded_values = encoded_values.reshape(n_ref_times, self.n_windows_per_forecast, -1)
            var_length = encoded_values.shape[-1]
            var_end += var_length * self.window_length
            var_indices['lead_time'] = ((var_start, var_start+1))
            var_start = var_end
            self.feature_vectors = np.concatenate([self.feature_vectors, encoded_values], axis=-1)
        if 'time_of_day' in self.weather_variables:
            reference_times = self.dataset['reference_time'].values
            horizon_timedelta = np.arange(0, self.n_windows_per_forecast, dtype='timedelta64[h]')
            horizon_reference_times = reference_times.reshape(-1, 1) + horizon_timedelta.reshape(1, -1)
            hour_of_day = (horizon_reference_times.astype('datetime64[h]') - horizon_reference_times.astype(
                'datetime64[D]')).astype(int)
            var_def = self.variable_definitions['time_of_day']
            encoded_values = var_def.encode(hour_of_day)
            encoded_values = encoded_values.reshape(n_ref_times, self.n_windows_per_forecast, - 1)
            var_length = encoded_values.shape[-1]
            var_end += var_length * self.window_length
            var_indices['time_of_day'] = ((var_start, var_start + 1))
            var_start = var_end
            self.feature_vectors = np.concatenate([self.feature_vectors, encoded_values], axis=-1)
        self.windows = self.feature_vectors.reshape(n_ref_times*self.n_windows_per_forecast, -1)
        self.variable_index = var_indices
        start_production_indices = self.dataset['production_index'].values
        # Each production index refers to the start of the forecast. To get all production for the forecast, we add
        # the horizon plus target lag
        production_indices = start_production_indices.reshape(-1, 1) + (np.arange(self.n_windows_per_forecast) + self.production_offset).reshape(1, -1)
        production_values = self.dataset[self.production_variable].isel(production_time=production_indices.flatten()).values

        var_def = self.variable_definitions[self.production_variable]
        encoded_targets = var_def.encode(production_values)
        self.targets = encoded_targets

    def __getitem__(self, item):
        data = dict(x=self.windows[item],
                    y=self.targets[item])
        if self.include_variable_index:
            data['variable_index'] = self.variable_index
        return data


class MultiSiteDataset(object):
    def __init__(self, datasets, *, window_length, production_offset, horizon=None,
                 weather_variables=None, production_variable='site_production'):
        self.datasets = [SiteDatasetOld(dataset_path=d, window_length=window_length, production_offset=production_offset,
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


DEFAULT_VARIABLE_CONFIG = VariableConfig(production_variable={
    'DWD_ICON-EU':  'site_production',
    "FMI_HIRLAM": 'site_production',
    "NCEP_GFS":  'site_production',
    "MetNo_MEPS": 'site_production',
}, variable_definitions={
    'DWD_ICON-EU': {'T': Variable('T'),
                    'U': Variable('U'),
                    'V': Variable('V'),
                    'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                       one_hot_encode=True),
                    'r': Variable('r'),
                    'lead_time': Variable('lead_time'),
                    'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                                       mapping={i: i for i in range(24)},
                                                       one_hot_encode=True),
                    'site_production': Variable('site_production'),
                    },
    "FMI_HIRLAM": {
        "Temperature": Variable("Temperature"),
        "WindUMS": Variable("WindUMS"),
        "WindVMS": Variable("WindVMS"),
        'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                           one_hot_encode=True),
        'r': Variable('r'),
        'lead_time': Variable('lead_time'),
        'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                           mapping={i: i for i in range(24)},
                                           one_hot_encode=True),
        'site_production': Variable('site_production'),
    },
    "NCEP_GFS": {'WindUMS_Height': Variable('WindUMS_Height'),
                 'WindVMS_Height': Variable('WindVMS_Height'),
                 'Temperature_Height': Variable('Temperature_Height'),
                 'PotentialTemperature_Sigma': Variable('PotentialTemperature_Sigma'),
                 'WindGust': Variable('WindGust'),
                 'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                    one_hot_encode=True),
                 'r': Variable('r'),
                 'lead_time': Variable('lead_time'),
                 'time_of_day': CategoricalVariable('time_of_day', levels=np.arange(24),
                                                    mapping={i: i for i in range(24)},
                                                    one_hot_encode=True),
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
        'phi_z': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                           one_hot_encode=True),
        'r_z': Variable('r'),
        'phi_10m': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                           one_hot_encode=True),
        'r_10m': Variable('r'),
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
                   'phi_10m', 'r_10m', 'phi_z', 'r_z', 'lead_time', 'time_of_day']
}
)


def main():
    import argparse
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('site_datafile', type=Path)
    args = parser.parse_args()


    dataset_config = DatasetConfig(horizon=30,
                                  window_length=7,
                                  production_offset=3,
                                  include_variable_index=True,
                                  use_cache=False,
                                  variable_config=DEFAULT_VARIABLE_CONFIG)

    site_dataset = SiteDataset(dataset_path=args.site_datafile, dataset_config=dataset_config)
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


