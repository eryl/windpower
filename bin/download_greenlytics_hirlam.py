import argparse
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import requests
import json
import xarray as xa
import time
from tqdm import trange, tqdm

dwd_height_table = {
    52: 883.557,
    53: 719.514,
    54: 570.373,
    55: 436.493,
    56: 318.336,
    57: 216.516,
    58: 131.880,
    59: 65.677,
    60: 20.000,
    61: 0.000,
}



def main():
    parser = argparse.ArgumentParser(description='Script for downloading NWP data from the Greenlytics API')
    parser.add_argument('metadata_file', help='CSV File containing the metadata', type=Path)
    parser.add_argument('sites_file', help='File containing the LPs of sites to download', type=Path)
    parser.add_argument('data_dir', help='Directory to store data to', type=Path)
    parser.add_argument('--api-key', help='Path to the api key file', default='../api_key')
    parser.add_argument('--rate-limit', help='Maximum number of requests per minute', type=int, default=5)
    parser.add_argument('--overwrite', help='If set, overwrite existing dataset files', action='store_true')

    args = parser.parse_args()

    with open(args.api_key) as fp:
        api_key = fp.readline().strip()

    metadata = pd.read_excel(args.metadata_file).dropna(thresh=15)  # The metadata file is 16 columns, one have some missing vnalues which doesn't matter
    with open(args.sites_file) as fp:
        sites = [site.strip() for site in fp if site.strip()]

    coordinates = set()
    for site in sites:
        site_row = metadata[metadata['LP'] == site]
        lat = float(site_row['Lat'])
        lon = float(site_row['Lon'])
        coordinates.add((lat, lon))

    for lat, lon in tqdm(sorted(coordinates), desc='Coordinates'):
        download_data(args.data_dir, api_key, lat, lon)


def download_data(data_dir, api_key, lat, lon, ref_times_per_request=1e5, freq=3, rate_limit=5,
                  start_date=datetime.datetime(2019, 1, 1, 0), end_date=None, overwrite=False):
    params = {
        'model': 'FMI_HIRLAM',
        # We trim away one hour, to not make overlapping requests
        'coords': {'latitude': [lat], 'longitude': [lon]},
        'variables': ['Temperature', 'WindUMS', 'WindVMS'],
        'freq': '6H',
        # 'as_dataframe': 'True',
        # 'as_dataframe': 'False',
    }

    if end_date is None:
        end_date = datetime.datetime.now()
    endpoint_url = "https://api.greenlytics.io/weather/v1/get_nwp"
    headers = {"Authorization": api_key}

    frequency = datetime.timedelta(hours=freq)

    n_ref_times = (end_date - start_date)//frequency
    data_dir.mkdir(parents=True, exist_ok=True)
    time_format = '%Y-%m-%d %H'

    n_requests = int(np.ceil(n_ref_times / ref_times_per_request))

    request_start = start_date

    seconds_per_request = 60 / rate_limit
    tm1 = time.time()
    for i in trange(n_requests, desc="Requests"):
        request_end = request_start + frequency*ref_times_per_request
        if request_end > end_date:
            request_end = end_date

        params['start_date'] = request_start.strftime(time_format)
        params['end_date'] = (request_end - datetime.timedelta(hours=1)).strftime(time_format)

        ## The rate limit of the API seems to only care about time between a request is finnished and a new one is
        # issued, not acually requests per minute. The code below assumes the limit is actual requests per minute
        #print(json.dumps(params, indent=2, sort_keys=True))
        # dt = time.time() - tm1
        # print("Before wait ", dt)
        # wait_time = seconds_per_request - dt
        # if wait_time > 0:
        #     time.sleep(wait_time)
        #
        # dt = time.time() - tm1
        # print("After wait: ", dt)
        # tm1 = time.time()

        time.sleep(seconds_per_request)  # For now, we just sleep enough to pass the server rate limiter
        response = requests.get(endpoint_url, headers=headers, params={'query_params': json.dumps(params)})
        response.raise_for_status()

        ds = xa.Dataset.from_dict(json.loads(response.text))
        ds['reference_time'] = ds['reference_time'].values.astype('datetime64[ns]')
        ds['valid_time'] = ds['valid_time'].astype(np.int32)
        file_name = data_dir / '{}_{},{}_{}--{}.nc'.format(params['model'], lat, lon,
                                                           request_start.strftime(time_format),
                                                           request_end.strftime(time_format))
        ds.to_netcdf(file_name)
        request_start = request_end


if __name__ == '__main__':
    main()