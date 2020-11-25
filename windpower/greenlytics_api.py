import os
import re
import datetime
import tempfile
import time
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import requests
import xarray as xa
from tqdm import trange, tqdm

@dataclass
class GreenLyticsModel:
    model_name: str
    identifier: str
    start_date: datetime
    variables: List[str]
    default_variables: List[str]
    base_frequency: int


GREENLYTICS_ENDPOINT_URL = "https://api.greenlytics.io/weather/v1/get_nwp"


MODELS = [
    GreenLyticsModel(model_name="DWD_ICON-EU", identifier="DWD_ICON-EU",
                     start_date=datetime.datetime(2019, 3, 5, 12),
                     base_frequency=3,
                     variables=[
                         "T",
                         "U",
                         "V",
                         "CLCT",
                         "CLCL",
                         "CLCM",
                         "CLCH",
                         "ASOB_S",
                         "ASWDIFD_S",
                         "ASWDIR_S",
                     ],
                     default_variables=[
                         "T",
                         "U",
                         "V",
                     ]),
    GreenLyticsModel(model_name="FMI_HIRLAM",
                     identifier="FMI_HIRLAM",
                     start_date=datetime.datetime(2019, 6, 24, 6),
                     base_frequency=6,
                     variables= [
                         "Temperature",
                         "WindUMS",
                         "WindVMS",
                         "TotalCloudCover",
                         "LowCloudCover",
                         "MediumCloudCover",
                         "HighCloudCover",
                         "RadiationGlobalAccumulation",
                     ],
                     default_variables=[
                         "Temperature",
                         "WindUMS",
                         "WindVMS",
                     ],),
    GreenLyticsModel(model_name="NCEP_GFS",
                     identifier="NCEP_GFS",
                     start_date=datetime.datetime(2019, 6, 24, 6),
                     base_frequency=6,
                     variables= [
                         "WindGust",
                         "WindUMS_Height",
                         "WindVMS_Height",
                         "WindUMS_Isobar",
                         "WindVMS_Isobar",
                         "StormMotionU_Height",
                         "StormMotionV_Height",
                         "StormRelativeHelicity_Height",
                         "SurfacePressure",
                         "PressureReducedMSL",
                         "RelativeHumidity_Isobar",
                         "RelativeHumidity_Height",
                         "PrecipitableWater",
                         "SurfacePrecipitationRate",
                         "SurfacePrecipitationRateAvg",
                         "SurfaceTotalPrecipitation",
                         "SurfaceSnowDepth",
                         "SurfaceWaterEqAccSnowDepth",
                         "Temperature_Height",
                         "PotentialTemperature_Sigma",
                         "SoilMoisture_Depth",
                         "SoilTemperature_Depth",
                         "PlanetaryBoundaryLayer_Height",
                         "CloudCover_Isobar",
                         "SurfaceRadiationShortWaveDownAvg",
                         "SurfaceRadiationShortWaveUpAvg",
                         "SurfaceRadiationLongWaveDownAvg",
                         "SurfaceLatentHeatNetFluxAvg",
                         "SurfaceSensibleHeatNetFluxAvg",
                     ],
                     default_variables=[
                         "WindUMS_Height",
                         "WindVMS_Height",
                         "Temperature_Height",
                     ]),
    GreenLyticsModel(model_name="MetNo_MEPS",
                     identifier="MetNo_MEPS",
                     start_date=datetime.datetime(2018, 10, 1, 0),
                     base_frequency=6,
                     variables= [
                         "x_wind_10m",
                         "y_wind_10m",
                         "x_wind_z",
                         "y_wind_z",
                         "air_pressure_at_sea_level",
                         "air_temperature_0m",
                         "air_temperature_2m",
                         "air_temperature_z",
                         "relative_humidity_2m",
                         "relative_humidity_z",
                         "cloud_area_fraction",
                         "low_type_cloud_area_fraction",
                         "medium_type_cloud_area_fraction",
                         "high_type_cloud_area_fraction",
                         "integral_of_rainfall_amount_wrt_time",
                         "integral_of_surface_net_downward_shortwave_flux_wrt_time",
                     ],
                     default_variables=[
                         "x_wind_10m",
                         "y_wind_10m",
                         "x_wind_z",
                         "y_wind_z",
                     ]),
    GreenLyticsModel(model_name="ECMWF_EPS-CF",
                     identifier="ECMWF_EPS-CF",
                     start_date=datetime.datetime(1992, 11, 24, 12),
                     base_frequency=6,
                     variables= [
                         "u10", # U-component of wind in meters per second (m.s-2) at 10 meters above the surface (GRIB variable documentation)
                         "v10", # V-component of wind in meters per second (m.s-2) at 10 meters above the surface (GRIB variable documentation)
                         "u100", # U-component of wind in meters per second (m.s-2) at 100 meters above the surface (GRIB variable documentation)
                         "v100", # V-component of wind in meters per second (m.s-2) at 100 meters above the surface (GRIB variable documentation)
                         "u200", # U-component of wind in meters per second (m.s-2) at 200 meters above the surface (GRIB variable documentation)
                         "v200", # V-component of wind in meters per second (m.s-2) at 200 meters above the surface (GRIB variable documentation)
                         "i10fg", # Instantaneous wind gusts in meters per second (m.s-1) at 10 meters above the surface (GRIB variable documentation)
                         "t2m", # Temperature in Kelvins (K) at 2 meters above the surface (GRIB variable documentation)
                         "d2m", # Dewpoint temperature in Kelvins (K) at 2 meters above the surface (GRIB variable documentation)
                         "tav300", # Average potential temperature in degrees Celsius (°C) in the upper 300m (GRIB variable documentation)
                         "msl", # Mean sea level pressure in Pascals (Pa) (GRIB variable documentation)
                         "tcc", # Total cloud cover in proportion (0-1) (GRIB variable documentation)
                         "lcc", # Low cloud cover in proportion (0-1) (GRIB variable documentation)
                         "mcc", # Medium cloud cover in proportion (0-1) (GRIB variable documentation)
                         "hcc", # High cloud cover in proportion (0-1) (GRIB variable documentation)
                         "dsrp", # Direct solar radiation in Joules per square meter (J.m-2) (GRIB variable documentation)
                         "uvb", # Downward UV radiation in Joules per square meter (J.m-2) at the surface (GRIB variable documentation)
                         "tp", # Total precipitation in meters (m) (GRIB variable documentation)
                         "ilspf", # Instantaneous large-scale surface precipitation fraction (GRIB variable documentation)
                     ],
                     default_variables= [
                         "u10",
                         "v10",
                         "u100",
                         "v100",
                         #"u200",
                         #"v200",
                         "i10fg",
                         "t2m",
                     ]),
    GreenLyticsModel(model_name="ECMWF_HRES",
                     identifier="ECMWF_HRES",
                     start_date=datetime.datetime(2019, 1, 1, 0, 0),
                     base_frequency=12,
                     variables= [
                         "u10", # U-component of wind in meters per second (m.s-2) at 10 meters above the surface (GRIB variable documentation)
                         "v10", # V-component of wind in meters per second (m.s-2) at 10 meters above the surface (GRIB variable documentation)
                         "u100", # U-component of wind in meters per second (m.s-2) at 100 meters above the surface (GRIB variable documentation)
                         "v100", # V-component of wind in meters per second (m.s-2) at 100 meters above the surface (GRIB variable documentation)
                         "u200", # U-component of wind in meters per second (m.s-2) at 200 meters above the surface (GRIB variable documentation)
                         "v200", # V-component of wind in meters per second (m.s-2) at 200 meters above the surface (GRIB variable documentation)
                         "i10fg", # Instantaneous wind gusts in meters per second (m.s-1) at 10 meters above the surface (GRIB variable documentation)
                         "t2m", # Temperature in Kelvins (K) at 2 meters above the surface (GRIB variable documentation)
                         "d2m", # Dewpoint temperature in Kelvins (K) at 2 meters above the surface (GRIB variable documentation)
                         "tav300", # Average potential temperature in degrees Celsius (°C) in the upper 300m (GRIB variable documentation)
                         "msl", # Mean sea level pressure in Pascals (Pa) (GRIB variable documentation)
                         "tcc", # Total cloud cover in proportion (0-1) (GRIB variable documentation)
                         "lcc", # Low cloud cover in proportion (0-1) (GRIB variable documentation)
                         "mcc", # Medium cloud cover in proportion (0-1) (GRIB variable documentation)
                         "hcc", # High cloud cover in proportion (0-1) (GRIB variable documentation)
                         "dsrp", # Direct solar radiation in Joules per square meter (J.m-2) (GRIB variable documentation)
                         "uvb", # Downward UV radiation in Joules per square meter (J.m-2) at the surface (GRIB variable documentation)
                         "tp", # Total precipitation in meters (m) (GRIB variable documentation)
                         "ilspf", # Instantaneous large-scale surface precipitation fraction (GRIB variable documentation)
                     ],
                     default_variables= [
                         "u10",
                         "v10",
                         "u100",
                         "v100",
                         #"u200",
                         #"v200",
                         "i10fg",
                         "t2m",
                     ]),
]

MODEL_MAP = {model.model_name: model for model in MODELS}

def check_params(model_name, variables, freq, start_date):
    """Check that the chosen variables and frequency is ok for the selected model_name"""
    model = MODEL_MAP[model_name]
    base_freq = model.base_frequency
    if freq % base_freq != 0:
        raise ValueError(f"Invalid frequency for {model_name}, frequency should be a multiple of {base_freq}.")

    model_valid_variables = model.variables
    for var in variables:
        invalid_variables = []
        if var not in model_valid_variables:
            invalid_variables.append(var)
        if invalid_variables:
            raise ValueError(f"Invalid variables for {model_name}: {invalid_variables}")

    if start_date < model.start_date:
        raise ValueError(f"Invalid start date {start_date} for model_name {model_name}, "
                         f"earliest possible date is {model.start_date}")




def download_coords(dest, coordinates, model_name, variables, api_key,
                    start_date=None, end_date=None, freq=None,
                    ref_times_per_request=1e5, rate_limit=5, overwrite=False,
                    coords_per_request=70):
    model = MODEL_MAP[model_name]
    # Set up default values
    if not variables:
        variables = model.default_variables
        print(f"No variables specified. Using default variables {variables}.")
    if not freq:
        freq = model.base_frequency

    if start_date is None:
        start_date = model.start_date
    elif not isinstance(start_date, datetime.datetime):
        try:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H')
        min_start_date = model.start_date
        if start_date < min_start_date:
            print(f"Invalid earliest start date for {model_name}, setting {start_date} to {min_start_date}")
            start_date = min_start_date

    if end_date is None:
        end_date = datetime.datetime.now()
    elif not isinstance(end_date, datetime.datetime):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    check_params(model_name, variables, freq, start_date)
    coord_chunks = []
    sorted_coords = list(sorted(coordinates, key=lambda x: (x['latitude'], x['longitude'])))
    n_coord_chunks = int(np.ceil(len(coordinates)/coords_per_request))
    for i in range(n_coord_chunks):
        start_coord = i*coords_per_request
        end_coord = start_coord + coords_per_request
        coord_chunks.append(sorted_coords[start_coord:end_coord])

    for coord_chunk in tqdm(coord_chunks, desc='Coordinates'):
        lats = [coord_dict['latitude'] for coord_dict in coord_chunk]
        lons = [coord_dict['longitude'] for coord_dict in coord_chunk]
        download_data(dest, api_key, model, variables, lats, lons,
                      start_date=start_date, end_date=end_date,
                      freq=freq, ref_times_per_request=ref_times_per_request,
                      rate_limit=rate_limit, overwrite=overwrite)


def get_missing_ref_times(dest, coords_pattern, start_date: datetime.datetime, end_date: datetime.datetime, freq_dt: datetime.timedelta, overlap=0.99):
    coords_files = list(dest.glob(coords_pattern + '*'))

    target_ref_times = set()
    ref_time = start_date
    while ref_time < end_date:
        target_ref_times.add(ref_time)
        ref_time += freq_dt

    epoch = np.datetime64('1970-01-01T00:00:00Z')
    np_s = np.timedelta64(1, 's')

    existing_ref_times = set()
    for coord_file in coords_files:
        with xa.open_dataset(coord_file) as ds:
            ref_times = ds['reference_time'].values
            timestamps = (ref_times - epoch) / np_s
            existing_ref_times.update([datetime.datetime.utcfromtimestamp(ts) for ts in timestamps])

    return target_ref_times - existing_ref_times






def download_data(dest, api_key, model: GreenLyticsModel, variables, lats, lons, ref_times_per_request=1e5, freq=6, rate_limit=60,
                  start_date=None, end_date=None, overwrite=False, use_direct_url=False):
    if use_direct_url:
        output_format = 'netcdf_url'
    else:
        output_format = 'json_xarray'
    headers = {"Authorization": api_key}
    base_params = {
        'model': model.identifier,
        'type': 'points',
        'coords': {'latitude': lats, 'longitude': lons, 'valid_time': list(range(0, 48))},
        'variables': variables,
        #'freq': '{}H'.format(freq),
        # 'as_dataframe': True,
        # 'as_dataframe': False,
        'output_format': output_format
    }

    frequency = datetime.timedelta(hours=freq)
    n_ref_times = (end_date - start_date)//frequency
    dest.mkdir(parents=True, exist_ok=True)
    time_format = '%Y-%m-%d %H'
    #coords_filename_part = f'{model_name}_{lat},{lon}'


    #missing_ref_times = get_missing_ref_times(dest, coords_filename_part, start_date, end_date, frequency)
    # if len(missing_ref_times)/n_ref_times < 0.05:
    #     print(f'Missing reference times is less than 5%, skipping (lat, lon): {lat}, {lon}')
    #     return

    n_requests = int(np.ceil(n_ref_times / ref_times_per_request))

    request_start = start_date

    seconds_per_request = 60 / rate_limit
    tm1 = time.time()
    for i in trange(n_requests, desc="Requests"):
        request_end = request_start + frequency*ref_times_per_request
        if request_end > end_date:
            request_end = end_date
        params = dict()
        params.update(base_params)
        params['start_date'] = request_start.strftime(time_format)
        params['end_date'] = (request_end - datetime.timedelta(hours=freq)).strftime(time_format)

        #print(json.dumps(params, indent=2, sort_keys=True))
        dt = time.time() - tm1
        wait_time = seconds_per_request - dt
        if wait_time > 0:
            time.sleep(wait_time)
        dt = time.time() - tm1
        #print("After wait: ", dt)
        tm1 = time.time()

        print(f'endpoint_url = "{GREENLYTICS_ENDPOINT_URL}"')
        print(f'headers = {{"Authorization": "{api_key}"}}')
        print(f'params = {params}')

        print("Making request with params: {}".format(json.dumps(params)))

        # request_params = {'query_params': json.dumps(params)}
        # print(f"Headers: {headers}\nParams: {request_params}")
        # req = requests.Request('GET', GREENLYTICS_ENDPOINT_URL, headers=headers, params=request_params).prepare()
        # print('{}\n{}\r\n{}\r\n\r\n{}'.format(
        #     '-----------START-----------',
        #     req.method + ' ' + req.url,
        #     '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        #     req.body,
        # ))
        # s = requests.Session()
        # response = s.send(req)
        response = requests.post(GREENLYTICS_ENDPOINT_URL,
                                  headers=headers,
                                  json={'query_params': params})#, content_type='application/json')
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Reguest header: {headers}, params: {params}")
            raise e
        if output_format == 'json_xarray':
            ds = xa.Dataset.from_dict(json.loads(response.text))
        else:
            netcdf_url = json.loads(response.text)['file']
            print(f"Downloading data from {netcdf_url}")
            with requests.get(netcdf_url, stream=True) as r:
                r.raise_for_status()
                fd, tmp_file = tempfile.mkstemp()
                with open(fd, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        # if chunk:
                        f.write(chunk)
                ds = xa.open_dataset(tmp_file)
                os.remove(tmp_file)

        ds['reference_time'] = ds['reference_time'].values.astype('datetime64[ns]')
        ds['valid_time'] = ds['valid_time'].astype(np.int32)
        ds.attrs['nwp_model'] = model.identifier
        response_start_date = min(ds['reference_time'].values).astype('datetime64[s]').tolist()
        response_end_date = max(ds['reference_time'].values).astype('datetime64[s]').tolist()
        if abs(request_start - response_start_date) > datetime.timedelta(days=1):
            print(f"Response and request times differ by more than a day: "
                  f"request_start: {request_start}, response_start: {response_start_date}. "
                  f"Request_end: {request_end}, response end: {response_end_date}."
                  f"Request params: {params}")

        for i, (lat,lon) in enumerate(zip(lats, lons)):
            # We try to make sure the latitudes and longitudes in the dataset are the same
            local_ds = ds.sel(point=i).drop_vars(['point'])
            ds_lat = local_ds['latitude']
            ds_lon = local_ds['longitude']
            if abs(ds_lat - lat) > 0.01 or abs(ds_lon - lon) > 0.01:
                print(f"Warning: Difference between lat,lon: {lat},{lon} and {ds_lat}, {ds_lon} is too great")

            # We update the filename with the actual datetime in the dataset
            file_name = dest / '{}_{},{}_{}--{}.nc'.format(params['model'], lat, lon,
                                                           response_start_date.strftime(time_format),
                                                           response_end_date.strftime(time_format))
            # Now slice out only the relevant dataset
            local_ds.to_netcdf(file_name)

        request_start = request_end


def parse_filename(f):
    """Return different parameters from a filename of a downloaded file
    :param f: Filename to parse
    :return A dictionary with the keys 'model_name', 'latitude', 'longitude'. If the file has a data range, the keys
            'start_date', 'end_date' are also present.
    """
    coord_fmt = r"\d+\.\d+"
    #model_fmt = '|'.join(MODELS)
    model_fmt = r'\w+'
    date_fmt = r"\d\d\d\d-\d\d-\d\d \d\d"
    date_pattern = r"({})_({}),({})_({})--({}).nc".format(model_fmt, coord_fmt, coord_fmt, date_fmt, date_fmt)
    nondate_pattern = r"({})_({}),({}).nc".format(model_fmt, coord_fmt, coord_fmt)
    m = re.match(date_pattern, f.name)
    if m is not None:
        model, latitude, longitude, start_date, end_date = m.groups()
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H')
        return dict(model=model, latitude=float(latitude), longitude=float(longitude), start_date=start_date,
                    end_date=end_date)
    else:
        m = re.match(nondate_pattern, f.name)
        if m is not None:
            model, latitude, longitude = m.groups()
            return dict(model=model, latitude=float(latitude), longitude=float(longitude))
    raise ValueError(f"Not a valid NWP dataset file name: {f}")
