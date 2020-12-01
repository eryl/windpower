"""Create site netCDF datasets, combining normalized production data and NWP data"""

import argparse
import re
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from windpower.greenlytics_api import GreenlyticsModelDataset
from windpower.dataset import SiteDatasetMetadata

bad_nwp_dates = defaultdict(Counter)

def main():
    parser = argparse.ArgumentParser(description="Create site datasets. A dataset will be created for each nwp dataset which matches the given site coordinates")
    parser.add_argument('site_metadata', type=Path)
    parser.add_argument('sites_file', type=Path)
    parser.add_argument('weather_dataset_dir', type=Path)
    parser.add_argument('production_data', type=Path)
    parser.add_argument('output_dir', help='Where to write site files. A subdirectory with the nwp model identifier will also be created.', type=Path)
    parser.add_argument('--coords-precision', help="Truncate coordinates to this precision", type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    metadata = pd.read_csv(args.site_metadata)
    with open(args.sites_file) as fp:
        sites = [site.strip() for site in fp if site.strip()]

    weather_coordinate_files = defaultdict(list)
    for f in args.weather_dataset_dir.glob('*.nc'):
        nwp_params = GreenlyticsModelDataset.fromstring(f.name)
        nwp_latitidue = round(nwp_params.latitude, args.coords_precision)
        nwp_longitude = round(nwp_params.longitude, args.coords_precision)
        weather_coordinate_files[(nwp_latitidue, nwp_longitude)].append((f, nwp_params.model))

    production_data = pd.read_csv(args.production_data, parse_dates=True, index_col=0, header=0)
    production_data.index = production_data.index.astype('datetime64[ns]')  #parse_dates adds timezone info, which is incompatible with the xarray conversion
    capacities = {str(row['LP']): row['Maxeffekt'] for i, row in metadata.iterrows()}
    for site in tqdm(sites):
        site_row = metadata[metadata['LP'] == int(site)]
        lat = round(float(site_row['Lat']), args.coords_precision)
        lon = round(float(site_row['Lon']), args.coords_precision)
        site_production = production_data[site].dropna()
        site_capacity = capacities[site]
        normalized_site_production = (site_production / site_capacity).clip(0, 1)
        normalized_site_production_dataarray = xr.DataArray(normalized_site_production)

        for weather_file, weather_model in weather_coordinate_files[(lat, lon)]:
            site_dataset_metadata = SiteDatasetMetadata(site_id=site, nwp_model=weather_model)
            output_dir = args.output_dir / weather_model.identifier
            output_dir.mkdir(exist_ok=True, parents=True)
            site_data_path = output_dir / f'{str(site_dataset_metadata)}.nc'

            if site_data_path.exists() and not args.overwrite:
                print(f"Skipping site {site} since the file exists")

            site_dataset = xr.open_dataset(weather_file)
            site_dataset['site_production'] = normalized_site_production_dataarray
            site_dataset.attrs['site_id'] = site
            site_dataset.attrs['capacity'] = site_capacity
            site_dataset.attrs['latitude'] = lat
            site_dataset.attrs['longitude'] = lon
            try:
                site_dataset = setup_xref(site_dataset)
            except ValueError:
                print(f"Can't make site dataset for {site} with weather file {weather_file}, the date ranges do not overlap")
                continue
            except KeyError as e:
                print(f"Can't make site dataset for {site} with weather file {weather_file}, received key error {e}")
                continue

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
    if end_time < start_time:
        raise ValueError("There is no overlap between production time and reference time, can't create dataset")

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
