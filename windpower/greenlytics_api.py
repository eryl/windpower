import datetime
import time
import json

import numpy as np
import requests
import xarray as xa
from tqdm import trange, tqdm


GREENLYTICS_ENDPOINT_URL = "https://api.greenlytics.io/weather/v1/get_nwp"
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
    if model == "DWD_ICON-EU":
        return datetime.datetime(2019, 3, 5, 9)
    elif model == "FMI_HIRLAM":
        return datetime.datetime(2019, 6, 24, 6)
    elif model == "NCEP_GFS":
        return datetime.datetime(2019, 6, 24, 6)
    elif model == "MetNo_MEPS":
        return datetime.datetime(2018, 10, 1, 0)
    else:
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
        #print(f"Headers: {headers}\nParams: {request_params}")
        req = requests.Request('GET', GREENLYTICS_ENDPOINT_URL, headers=headers, params=request_params).prepare()
        # print('{}\n{}\r\n{}\r\n\r\n{}'.format(
        #     '-----------START-----------',
        #     req.method + ' ' + req.url,
        #     '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        #     req.body,
        # ))
        s = requests.Session()
        response = s.send(req)
        response.raise_for_status()

        ds = xa.Dataset.from_dict(json.loads(response.text))
        ds['reference_time'] = ds['reference_time'].values.astype('datetime64[ns]')
        ds['valid_time'] = ds['valid_time'].astype(np.int32)
        ds['nwp_model'] = model
        ds.to_netcdf(file_name)
        request_start = request_end
