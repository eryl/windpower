"""Create site netCDF datasets, combining normalized production data and NWP data"""

import argparse
import re
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

bad_nwp_dates = defaultdict(Counter)

def main():
    parser = argparse.ArgumentParser(description="Test the dataset")
    parser.add_argument('site_metadata', type=Path)
    parser.add_argument('sites_file', type=Path)
    parser.add_argument('weather_dataset_dir', type=Path)
    parser.add_argument('production_data', type=Path)
    parser.add_argument('output_dir', type=Path)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)

    #weather_dataset = xr.open_dataset(args.weather_dataset)
    metadata = pd.read_excel(args.site_metadata).dropna(thresh=15)  # The metadata file is 16 columns, one have some missing vnalues which doesn't matter
    with open(args.sites_file) as fp:
        sites = [site.strip() for site in fp if site.strip()]

    pattern = re.compile(r'(DWD_ICON-EU|NCEP_GFS)_(\d*\.\d*),(\d*\.\d*)_.*\.nc')
    weather_coordinate_files = dict()
    for f in args.weather_dataset_dir.glob('*.nc'):
        m = re.match(pattern, f.name)
        if m is not None:
            model, lat, lon = m.groups()
            weather_coordinate_files[(float(lat), float(lon))] = (f, model)

    production_data = pd.read_csv(args.production_data, sep=',', parse_dates=True, index_col=0, header=0, skiprows=[1])
    production_data.index.name = 'production_time'
    production_data.index = production_data.index.astype('datetime64[ns]')  #parse_dates adds timezone info, which is incompatible with the xarray conversion
    capacities = pd.read_csv(args.production_data, sep=',', parse_dates=True, index_col=0, header=0, nrows=1)

    for site in tqdm(sites):
        site_row = metadata[metadata['LP'] == site]
        lat = float(site_row['Lat'])
        lon = float(site_row['Lon'])
        try:
            weather_file, weather_model = weather_coordinate_files[(lat, lon)]
        except KeyError:
            print(f"No weather data found for site at {lat},{lon}")
            continue
        site_production = production_data[site].dropna()
        site_capacity = capacities[site].to_numpy()
        normalized_site_production = (site_production / site_capacity).clip(0, 1)
        normalized_site_production_dataarray = xr.DataArray(normalized_site_production)
        site_dataset = xr.open_dataset(weather_file)
        site_dataset['site_production'] = normalized_site_production_dataarray
        site_dataset.attrs['site_id'] = site
        site_dataset.attrs['capacity'] = site_capacity[0]
        site_dataset.attrs['latitude'] = lat
        site_dataset.attrs['longitude'] = lon
        site_dataset = setup_xref(site_dataset)
        site_data_path = args.output_dir / '{}_{}.nc'.format(site, weather_model)
        site_dataset.to_netcdf(site_data_path)
    with open('/tmp/bad_dwd_dates.txt', 'w') as fp:
        for date, variable_counts in bad_nwp_dates.items():
            fp.write('{},{}\n'.format(date, ','.join(['{}:{}'.format(v,c) for v,c in sorted(variable_counts.items())])))


def setup_xref(dataset):
    # We need to figure out what times we can actually predict for, which will be the intersection of the time
    # intervals of weather predictions and production
    forecast_times = dataset['reference_time'].values
    production_times = dataset['production_time'].values

    # This holds pairs of the starting index of valid forecasts in the "reference_time" dimension and the corresponding
    # start index of production in the "production_time" dimension.
    forecast_production_offsets = []

    start_time = max(production_times.min(), forecast_times.min())
    end_time = min(production_times.max(), forecast_times.max())

    dataset = dataset.sel(reference_time=slice(start_time, end_time), production_time=slice(start_time, end_time))
    forecast_times = dataset['reference_time'].values
    production_times = dataset['production_time'].values
    reference_time_to_production_time_index = np.full(len(forecast_times), -1, dtype=np.int32)

    production_time_i = 0
    for forecast_time_i, forecast_time in enumerate(forecast_times):
        if forecast_time < start_time:
            continue
        elif forecast_time >= end_time:
            break

        # The forecast starts at reference_time_i, we need to figure out at which index the corresponding
        # production start. This assumes production times are sorted
        while production_times[production_time_i] < forecast_time:
            production_time_i += 1

        # We're actually seeing data quality issues where some forecasts have NaN values. For now just drop those
        window_data = dataset.isel(reference_time=forecast_time_i)

        nan_variables = []
        for var, dataarray in window_data.items():
            if dataarray.isnull().any():
                nan_variables.append(var)
        if nan_variables:
            for var in nan_variables:
                bad_nwp_dates[forecast_time][var] += 1
        else:
            reference_time_to_production_time_index[forecast_time_i] = production_time_i

    #dataset['forecast_production_offsets'] = xr.DataArray(forecast_production_offsets)
    dataset['production_index'] = xr.DataArray(reference_time_to_production_time_index, dims=('reference_time',))
    return dataset



if __name__ == '__main__':
    main()
