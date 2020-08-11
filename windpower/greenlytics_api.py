import re
import datetime
import time
import json

import numpy as np
import requests
import xarray as xa
from tqdm import trange, tqdm


GREENLYTICS_ENDPOINT_URL = "https://api.greenlytics.io/weather/v1/get_nwp"
MODELS = ["DWD_ICON-EU", "FMI_HIRLAM", "NCEP_GFS"]
VALID_VARIABLES = {
    "DWD_ICON-EU": [
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
    "FMI_HIRLAM": [
        "Temperature",
        "WindUMS",
        "WindVMS",
        "TotalCloudCover",
        "LowCloudCover",
        "MediumCloudCover",
        "HighCloudCover",
        "RadiationGlobalAccumulation",
    ],
    "NCEP_GFS": [
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
    "MetNo_MEPS": [
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
}

DEFAULT_VARIABLES = {
    "DWD_ICON-EU": [
        "T",
        "U",
        "V",
    ],
    "FMI_HIRLAM": [
        "Temperature",
        "WindUMS",
        "WindVMS",
    ],
    "NCEP_GFS": [
        "WindUMS_Height",
        "WindVMS_Height",
        "Temperature_Height",
    ],
    "MetNo_MEPS": [
        "x_wind_10m",
        "y_wind_10m",
        "x_wind_z",
        "y_wind_z",
    ],
}

MODEL_BASE_FREQUENCY = {
    "DWD_ICON-EU": 3,
    "FMI_HIRLAM": 6,
    "NCEP_GFS": 6,
    "MetNo_MEPS": 6,
}

MODEL_START_DATES = {
 "DWD_ICON-EU": datetime.datetime(2019, 3, 5, 9),
    "FMI_HIRLAM": datetime.datetime(2019, 6, 24, 6),
    "NCEP_GFS": datetime.datetime(2019, 6, 24, 6),
    "MetNo_MEPS": datetime.datetime(2018, 10, 1, 0),
}

def check_params(model, variables, freq):
    """Check that the chosen variables and frequency is ok for the selected model"""
    if model == "DWD_ICON-EU" and freq % 3 != 0:
        raise ValueError(f"Invalid frequency for {model}, frequency should be a multiple of 3.")
    elif (model == "NCEP_GFS" or model == "FMI_HIRLAM" or model == "MetNo_MEPS") and freq % 6 != 0:
        raise ValueError(f"Invalid frequency for {model}, frequency should be a multiple of 6.")

    model_valid_variables = VALID_VARIABLES[model]
    for var in variables:
        invalid_variables = []
        if var not in model_valid_variables:
            invalid_variables.append(var)
        if invalid_variables:
            raise ValueError(f"Invalid variables for {model}: {invalid_variables}")


def earliest_start_date(model):
    try:
        return MODEL_START_DATES[model]
    except KeyError:
        raise ValueError(f"No such model {model}")


def download_coords(dest, coordinates, model, variables, api_key,
                    start_date=None, end_date=None, freq=6,
                    ref_times_per_request=1e5, rate_limit=5, overwrite=False):

    if start_date is not None and not isinstance(start_date, datetime.datetime):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if end_date is not None and not isinstance(end_date, datetime.datetime):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    check_params(model, variables, freq)
    for coord_dict in tqdm(sorted(coordinates, key=lambda x: (x['latitude'], x['longitude'])), desc='Coordinate'):
        lat = coord_dict['latitude']
        lon = coord_dict['longitude']
        download_data(dest, api_key, model, variables, lat, lon,
                      start_date=start_date, end_date=end_date,
                      freq=freq, ref_times_per_request=ref_times_per_request,
                      rate_limit=rate_limit, overwrite=overwrite)

def download_data(dest, api_key, model, variables, lat, lon, ref_times_per_request=1e4, freq=6, rate_limit=5,
                  start_date=None, end_date=None, overwrite=False):
    if start_date is None:
        start_date = earliest_start_date(model)
    if end_date is None:
        end_date = datetime.datetime.now()

    headers = {"Authorization": api_key}
    base_params = {
        'model': model,
        'coords': {'latitude': [lat], 'longitude': [lon]},
        'variables': variables,
        'freq': '{}H'.format(freq),
        # 'as_dataframe': True,
        'as_dataframe': False,
    }

    frequency = datetime.timedelta(hours=freq)
    n_ref_times = (end_date - start_date)//frequency
    dest.mkdir(parents=True, exist_ok=True)
    time_format = '%Y-%m-%d %H'

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
        params['end_date'] = (request_end - datetime.timedelta(hours=1)).strftime(time_format)

        file_name = dest / '{}_{},{}_{}--{}.nc'.format(params['model'], lat, lon,
                                                       request_start.strftime(time_format),
                                                       request_end.strftime(time_format))
        if file_name.exists() and not overwrite:
            continue

        #print(json.dumps(params, indent=2, sort_keys=True))
        dt = time.time() - tm1
        wait_time = seconds_per_request - dt
        if wait_time > 0:
            time.sleep(wait_time)
        dt = time.time() - tm1
        #print("After wait: ", dt)
        tm1 = time.time()
        print("Making request with params: {}".format(json.dumps(params)))

        request_params = {'query_params': json.dumps(params)}
        print(f"Headers: {headers}\nParams: {request_params}")
        req = requests.Request('GET', GREENLYTICS_ENDPOINT_URL, headers=headers, params=request_params).prepare()
        print('{}\n{}\r\n{}\r\n\r\n{}'.format(
            '-----------START-----------',
            req.method + ' ' + req.url,
            '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
            req.body,
        ))
        s = requests.Session()
        response = s.send(req)
        response.raise_for_status()

        ds = xa.Dataset.from_dict(json.loads(response.text))
        ds['reference_time'] = ds['reference_time'].values.astype('datetime64[ns]')
        ds['valid_time'] = ds['valid_time'].astype(np.int32)
        ds.attrs['nwp_model'] = model
        response_start_date = min(ds['reference_time'].values).astype('datetime64[s]').tolist()
        response_end_date = max(ds['reference_time'].values).astype('datetime64[s]').tolist()
        if abs(request_start - response_start_date) > datetime.timedelta(days=1):
            print(f"Response and request times differ by more than a day: "
                  f"request_start: {request_start}, response_start: {response_start_date}. "
                  f"Request_end: {request_end}, response end: {response_end_date}."
                  f"Request params: {request_params}")

        # We update the filename with the actual datetime in the dataset
        file_name = dest / '{}_{},{}_{}--{}.nc'.format(params['model'], lat, lon,
                                                       response_start_date.strftime(time_format),
                                                       response_end_date.strftime(time_format))
        ds.to_netcdf(file_name)
        request_start = request_end


def parse_filename(f):
    """Return different parameters from a filename of a downloaded file
    :param f: Filename to parse
    :return A dictionary with the keys 'model', 'latitude', 'longitude'. If the file has a data range, the keys
            'start_date', 'end_date' are also present.
    """
    coord_fmt = r"\d+\.\d+"
    model_fmt = r"DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS|DWD_NCEP"
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
