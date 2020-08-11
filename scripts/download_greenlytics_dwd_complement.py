import argparse
import numpy as np
from pathlib import Path
import requests
import json
import xarray as xr
import time
from tqdm import trange, tqdm
from collections import defaultdict, Counter
import re
from dateutil.parser import parse

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

pattern = re.compile(r'(DWD_ICON-EU)_(\d*\.\d*),(\d*\.\d*)_20\d\d-\d\d-\d\d \d\d.nc')

def main():
    parser = argparse.ArgumentParser(description='Script for downloading NWP data from the Greenlytics API')
    parser.add_argument('coord_dirs', help='Directory containing data files', type=Path, nargs='+')
    parser.add_argument('--api-key', help='Path to the api key file', default='../api_key')
    parser.add_argument('--rate-limit', help='Maximum number of requests per minute', type=int, default=5)
    parser.add_argument('--overwrite', help='If set, overwrite existing dataset files', action='store_true')

    args = parser.parse_args()

    lat_lon_files = defaultdict(set)
    for d in args.coord_dirs:
        for f in d.glob('**/*.nc'):
            m = re.match(pattern, f.name)
            if m is not None:
                model, lat, lon = m.groups()
                lat_lon_files[lat, lon].add(f)

    with open(args.api_key) as fp:
        api_key = fp.readline().strip()

    for (lat, lon), files in tqdm(sorted(lat_lon_files.items()), desc='Coordinates'):
        download_data(files, api_key, lat, lon)

def download_data(files, api_key, lat, lon, ref_times_per_request=1, freq=3, rate_limit=5):
    data_dirs = Counter(f.parent for f in files)
    [(data_dir, n_files)] = data_dirs.most_common(1)

    params = {
        'model': 'DWD_ICON-EU',
        # We trim away one hour, to not make overlapping requests
        'coords': {'latitude': [lat], 'longitude': [lon]},
        'variables': ['T', 'U', 'V'],
        #'freq': '3H',
        # 'as_dataframe': 'True',
        # 'as_dataframe': 'False',
    }
    dates = set()
    for f in files:
        ds = xr.open_dataset(f)
        dates.update(ds['reference_time'].values)

    start_date = min(dates)
    end_date = max(dates)

    td = np.timedelta64(3, 'h')

    missing_dates = set()
    current_date = start_date
    while current_date < end_date:
        if current_date not in dates:
            missing_dates.add(current_date)
        current_date += td
    missing_dates = [parse(str(d)) for d in sorted(missing_dates)]

    endpoint_url = "https://api.greenlytics.io/weather/v1/get_nwp"
    headers = {"Authorization": api_key}

    time_format = '%Y-%m-%d %H'
    n_requests = int(np.ceil(len(missing_dates) / ref_times_per_request))

    seconds_per_request = 60 / rate_limit
    tm1 = time.time()
    for i in trange(n_requests, desc="Requests"):
        ref_datetimes = missing_dates[i*ref_times_per_request: (i+1)*ref_times_per_request]

        #params['start_date'] = request_start.strftime(time_format)
        #params['end_date'] = (request_end - datetime.timedelta(hours=1)).strftime(time_format)
        params['ref_datetimes'] = [ref_time.strftime(time_format) for ref_time in ref_datetimes]

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

        ds = xr.Dataset.from_dict(json.loads(response.text))
        ds['reference_time'] = ds['reference_time'].values.astype('datetime64[ns]')
        ds['valid_time'] = ds['valid_time'].astype(np.int32)

        for i, ref_time in enumerate(ds['reference_time']):
            ds_at_ref_time = ds.isel(reference_time = slice(i, i+1))
            reference_datetime = parse(str(ds_at_ref_time['reference_time'].values[0]))
            file_name = data_dir / '{}_{},{}_{}.nc'.format(params['model'], lat, lon, reference_datetime.strftime(time_format))
            ds_at_ref_time.to_netcdf(file_name)


if __name__ == '__main__':
    main()