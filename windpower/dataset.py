import pickle
from collections import defaultdict, Counter
import hashlib
import xarray as xr
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
from windpower.utils import load_module, sliding_window
from windpower.greenlytics_api import MODELS
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class VariableType(Enum):
    continuous = 1
    discrete = 2

class Variable(object):
    type = VariableType.continuous
    def __init__(self, name):
        self.name = name

    def encode(self, v):
        return v

    def decode(self, v):
        return v


class CategoricalVariable(Variable):
    type = VariableType.discrete
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
    include_variable_info: bool


@dataclass
class SplitConfig:
    outer_folds: int  # How many outer cross-validation folds to use
    inner_folds: int  # how many inner cross-validation folds to use. If 1, a single validation set is split of to pad the "outer" held out set
    split_padding: int # How many hours of padding between each split
    validation_ratio: float = 0.1  # How much data to split of to the validation set, will only be used if inner_folds=1
    outer_fold_idxs: Optional[List[int]] = None  # If set, only produce these exact folds (so needs to be less than outer_folds). E.g. if outer_folds = 10 and outer_fold_idxs=[9], only the last outer fold is produced
    inner_fold_idxs: Optional[List[int]] = None  # If set, only produce these exact folds (so needs to be less than inner_folds). E.g. if inner_folds = 10 and inner_fold_idxs=[9], only the last inner fold is produced



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


def get_nwp_model(dataset_path: Path):
    try:
        return get_nwp_model_from_path(dataset_path)
    except ValueError as e:
        with xr.open_dataset(dataset_path) as dataset:
            if 'nwp_model' in dataset.attrs:
                return dataset.attrs['nwp_model']
            else:
                raise e


def get_nwp_model_from_path(dataset_path: Path):
    import re
    #pattern = re.compile(r'\d+_(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).nc|.*(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).*.nc')
    model_pattern = '|'.join(MODELS)
    pattern = re.compile(r'.*(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS|DWD_NCEP|ECMWF_EPS-CF|DWD_ECMWF_NCEP).*.nc')
    m = re.match(pattern, dataset_path.name)
    if m is not None:
        (model,) = m.groups()
        return model
    else:
        raise ValueError(f"Not a dataset path: {dataset_path}")


def get_site_id(dataset_path: Path):
    import re
    # pattern = re.compile(r'\d+_(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).nc|.*(DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS).*.nc')
    pattern = re.compile(r'(\d+)_(DWD_NCEP|DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS|ECMWF_EPS-CF|DWD_ECMWF_NCEP).nc')
    m = re.match(pattern, dataset_path.name)
    if m is not None:
        site_id, model, = m.groups()
        return site_id
    else:
        print(f"Could not determine site id from path path: {dataset_path}, loading dataset")
        with xr.open_dataset(dataset_path) as ds:
            site_id = ds.attrs['site_id']  # This is the most
            return site_id


def get_reference_time(site_dataset_path: Path):
    with xr.open_dataset(site_dataset_path) as ds:
        return ds['reference_time'].values



def old_k_fold_split_reference_times(forecast_times, k, padding):

    fold_intervals = split_datetimes(forecast_times, k, padding)

    for i in range(len(fold_intervals)):
        fold_start, fold_end = fold_intervals[i]
        fold_times = forecast_times[fold_start:fold_end]
        remainder_folds = []

        # we should
        if i > 0:
            previous_fold_start, previous_fold_end = fold_intervals[i - 1]
            remainder_folds.append(forecast_times[:previous_fold_end + 1])
        if i < len(fold_intervals) - 1:
            next_fold_start, next_fold_end = fold_intervals[i + 1]
            remainder_folds.append(forecast_times[next_fold_start:])

        remainder_times = np.concatenate(remainder_folds)
        yield fold_times, remainder_times


def k_fold_split_reference_times(forecast_times, k, padding):
    forecast_times = np.sort(forecast_times)
    # This method will not necessarily give equal sized folds if there are holes in the timeline
    times_per_forecast = len(forecast_times) // k
    padding_dt = np.timedelta64(padding, 'h')
    for i in range(k):
        fold_times = forecast_times[i*times_per_forecast : (i+1)*times_per_forecast]
        # since the times are sorted we can take the first and last as start time and end time
        fold_start_time = fold_times[0]
        fold_end_time = fold_times[-1]
        remainder_times_before = [t for t in forecast_times[:i*times_per_forecast]
                                  if t <= fold_start_time - padding_dt]
        remainder_times_after = [t for t in forecast_times[(i+1)*times_per_forecast:]
                                 if t >= fold_end_time + padding_dt]
        remainder_times = np.array(remainder_times_before+remainder_times_after)
        if len(fold_times) < 1 or len(remainder_times) < 1:
            raise RuntimeError("Intervals can not be separated, padding is too large (likely dataset is too small)")
        yield fold_times, remainder_times


def distance_split_reference_times(test_reference_times, train_reference_time, validation_ratio, fold_padding):
    """
    Splits the train_reference times into a validation and training set, where the validation set pads the training set evenly on both sides (if the test set is contained in test set).
    :param test_reference_times: The test set to use as reference. This should be a contigous time periods
    :param train_reference_time:
    :param validation_ratio:
    :param fold_padding:
    :return:
    """
    min_test_time = test_reference_times.min()
    max_test_time = test_reference_times.max()

    # Find the reference times of the train dataset closest to thos of the test dataset, we only have to check the first and last time of the reference time dataset since it is contigous
    distance_to_test_min = np.abs(train_reference_time-min_test_time)
    distance_to_test_max = np.abs(train_reference_time-max_test_time)
    # Stack the two distances on top of eachother, pick the minimum one which tells us how close this reference time is to the test set
    distance_to_test = np.min(np.stack([distance_to_test_min, distance_to_test_max], axis=0), axis=0)
    idx_sorted_by_distance_to_test = np.argsort(distance_to_test)
    n_validation_reference_times = int(round(len(train_reference_time)*validation_ratio))
    validation_reference_time_indices = idx_sorted_by_distance_to_test[:n_validation_reference_times]
    train_reference_time_indices = idx_sorted_by_distance_to_test[n_validation_reference_times:]
    val_times = train_reference_time[validation_reference_time_indices]
    train_times = train_reference_time[train_reference_time_indices]
    # Now the question is what training reference times to keep. The valid times pads the test times, so no training
    # times are in the interval [val_time.min(), val_time.max()] and we can repeat the procedure above
    distance_to_valid_min = np.abs(train_times - val_times.min())
    distance_to_valid_max = np.abs(train_times - val_times.max())
    distance_to_valid = np.min(np.stack([distance_to_valid_min, distance_to_valid_max], axis=0), axis=0)
    # Only select the training time indices where the distance to valid is greater than the fold_padding
    train_times = train_times[distance_to_valid > np.timedelta64(fold_padding, 'h')]
    np.sort(val_times)
    np.sort(train_times)

    return val_times, train_times


def make_splits(reference_time, split_config: SplitConfig):
    splits = []
    for i, (test_reference_times, train_reference_time) in enumerate(k_fold_split_reference_times(reference_time,
                                                                                                  split_config.outer_folds,
                                                                                                  split_config.split_padding)):
        if (split_config.outer_fold_idxs is not None
              and i not in split_config.outer_fold_idxs):
            continue

        if split_config.inner_folds > 1:
            inner_settings = []
            for j, (validation_dataset_reference_times, fit_dataset_reference_times) in enumerate(k_fold_split_reference_times(train_reference_time,
                                                                                                                               split_config.inner_folds,
                                                                                                                               split_config.split_padding)):
                if (split_config.inner_fold_idxs is not None
                      and j not in split_config.inner_fold_idxs):
                    continue
                inner_settings.append((j, validation_dataset_reference_times, fit_dataset_reference_times))
            splits.append((i, test_reference_times, inner_settings))

        elif split_config.inner_folds == 1:
            # We will create a validation dataset which pads the inner fold.
            validation_reference_times, fit_reference_times = distance_split_reference_times(test_reference_times,
                                                                                             train_reference_time,
                                                                                             split_config.validation_ratio,
                                                                                             split_config.split_padding)
            splits.append((i, test_reference_times, validation_reference_times, fit_reference_times))
        else:
            splits.append((i, test_reference_times, train_reference_time))
    return splits


def make_all_site_splits(datasets: List[Path], output_dir: Path, split_config: SplitConfig):
    per_site_datasets = defaultdict(list)

    for dataset_path in datasets:
        site_id = get_site_id(dataset_path)
        per_site_datasets[site_id].append(dataset_path)

    for site_id, site_datasets in tqdm(per_site_datasets.items(), desc="Site", total=len(per_site_datasets)):
        # Now determine the reference time intersections
        make_site_splits(site_id, site_datasets, output_dir=output_dir, split_config=split_config)


def make_site_splits(site_id, dataset_paths: List[Path], output_dir: Path, split_config: SplitConfig):
    reference_times = None
    for dataset_path in dataset_paths:
        with xr.open_dataset(dataset_path) as ds:
            ds_reference_times = ds['reference_time'].values
            if reference_times is None:
                reference_times = set(ds_reference_times)
            else:
                reference_times.intersection_update(set(ds_reference_times))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'site_{site_id}_splits.pkl'
    splits = make_splits(list(sorted(reference_times)), split_config)
    with open(output_file, 'wb') as fp:
        pickle.dump(dict(splits=splits, split_config=split_config, site_id=site_id), fp)
    return output_file



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
        self.nwp_model = get_nwp_model(dataset_path)
        self.reference_time = reference_time
        self.variables_config = variables_config
        self.dataset_config = dataset_config
        self.weather_variables = variables_config.weather_variables[self.nwp_model]
        self.variable_definitions = variables_config.variable_definitions[self.nwp_model]
        self.production_variable = variables_config.production_variable[self.nwp_model]
        self.horizon = dataset_config.horizon
        self.window_length = dataset_config.window_length
        self.production_offset = dataset_config.production_offset
        self.include_variable_info = dataset_config.include_variable_info
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
        self.n_windows_per_forecast = (self.horizon - self.window_length) + 1
        self.len = len(self.dataset['reference_time'])*self.n_windows_per_forecast

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
        return self.len

    def make_memdataset(self):
        n_ref_times = len(self.dataset['reference_time'])
        n_valid_times = len(self.dataset['valid_time'])

        var_arrays = []
        var_info = dict()
        var_start = 0
        var_end = 0
        for var in self.weather_variables:
            if var == 'lead_time' or var == 'time_of_day':
                #Lead time and time of day are not stored in the dataset, they are derived from the reference time
                continue
            var_values = self.dataset[var].values  # Pick out the numpy array values
            if not (var_values.shape[0] == n_ref_times and var_values.shape[1] == n_valid_times):
                raise ValueError(f"Variable {var} has mismatching shape: {var_values.shape}")
            var_def = self.variable_definitions[var]
            encoded_values = var_def.encode(var_values)
            var_values = encoded_values.reshape(n_ref_times, n_valid_times, -1)
            var_arrays.append(var_values)
            var_length = var_values.shape[-1]
            # Since each variable is repeated window_length number of times, we set the var_index to this value here
            var_end += var_length*self.window_length
            var_info[var] = ((var_start, var_end, var_def.type))
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
            var_info['lead_time'] = ((var_start, var_start+1, var_def.type))
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
            var_info['time_of_day'] = ((var_start, var_start + 1, var_def.type))
            var_start = var_end
            self.feature_vectors = np.concatenate([self.feature_vectors, encoded_values], axis=-1)
        self.windows = self.feature_vectors.reshape(n_ref_times*self.n_windows_per_forecast, -1)
        self.variable_info = var_info
        start_production_indices = self.dataset['production_index'].values
        # Each production index refers to the start of the forecast. To get all production for the forecast, we add
        # the horizon plus target lag
        production_indices = start_production_indices.reshape(-1, 1) + (np.arange(self.n_windows_per_forecast) + self.production_offset).reshape(1, -1)
        production_values = self.dataset[self.production_variable].isel(production_time=production_indices.flatten()).values

        var_def = self.variable_definitions[self.production_variable]
        encoded_targets = var_def.encode(production_values)
        self.targets = encoded_targets

    def __getitem__(self, item):
        if not hasattr(self, 'windows'):
            self.make_memdataset()
        data = dict(x=self.windows[item],
                    y=self.targets[item])
        if self.include_variable_info:
            data['variable_info'] = self.variable_info
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
    "ECMWF_EPS-CF": 'site_production',
    'DWD_ECMWF_NCEP': 'site_production',
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
                     'time_of_day']
}
)

DEFAULT_DATASET_CONFIG = DatasetConfig(horizon=30,
                                       window_length=7,
                                       production_offset=3,
                                       include_variable_info=True,
                                       )

def main():
    import matplotlib.pyplot as plt
    start_ref_time = np.datetime64('2019-01-01', 'h')
    reference_times = start_ref_time + np.arange(6000)*np.timedelta64(1, 'h')
    artists = None
    for i, (test_times, full_train_times) in enumerate(k_fold_split_reference_times(reference_times, 10, 12)):
        valid_times, train_times = distance_split_reference_times(test_times, full_train_times, 0.1, 12)
        test_p = plt.scatter(test_times, np.zeros(len(test_times))+i, label='test', alpha=.3, c='green')
        valid_p = plt.scatter(valid_times, np.zeros(len(valid_times))+i, label='valid', alpha=.3, c='blue')
        train_p = plt.scatter(train_times, np.zeros(len(train_times))+i, label='train', alpha=.3, c='orange')
        artists = zip(*((test_p, 'test'), (valid_p, 'valid'), (train_p, 'train')))
    plt.legend(*artists)
    plt.show()



def main_old():
    import argparse
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('site_datafile', type=Path)
    args = parser.parse_args()


    dataset_config = DatasetConfig(horizon=30,
                                  window_length=7,
                                  production_offset=3,
                                  include_variable_info=True,
                                  )

    site_dataset = SiteDataset(dataset_path=args.site_datafile,
                               dataset_config=dataset_config,
                               variables_config=DEFAULT_VARIABLE_CONFIG)
    zero_std_vars = Counter()
    for i, (fold, remainder) in enumerate(site_dataset.k_fold_split(10)):
        for j, (inner_fold, inner_remainder) in enumerate(remainder.k_fold_split(10)):
            data = inner_fold[:]
            x = data['x']
            std = np.std(x, axis=0)
            zero_std_i, = np.where(std == 0)
            variable_index = data['variable_info']
            for i in zero_std_i:
                for var, (start, end, var_type) in variable_index.items():
                    if start <= i < end:
                        zero_std_vars[var] += 1
    print(zero_std_vars)




if __name__ == '__main__':
    main()


