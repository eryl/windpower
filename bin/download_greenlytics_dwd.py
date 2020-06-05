import argparse
import csv
from pathlib import Path

import windpower.greenlytics_api

MODEL = "DWD_ICON-EU"
DEFAULT_VARIABLES = [
    "T",
    "U",
    "V",
]

def main():
    parser = argparse.ArgumentParser(description='Script for downloading NWP data from the Greenlytics API')
    parser.add_argument('coords_file', help='CSV File containing the longituide '
                                            'and latitude pairs to download data for', type=Path)
    parser.add_argument('data_dir', help='Directory to store data to', type=Path)
    parser.add_argument('--variables', help="What variables to download",
                        nargs='+',
                        default=DEFAULT_VARIABLES,
                        choices=windpower.greenlytics_api.VALID_VARIABLES[MODEL])
    parser.add_argument('--start-date', help="Start downloading from this date. If not set, "
                                             "the earliest possible date will be used. Format should be '%Y-%m-%d'")
    parser.add_argument('--end-date', help="End downloading from this date. If not set, "
                                           "current datetime will be used. Format should be '%Y-%m-%d'")
    parser.add_argument('--api-key', help='Path to the api key file', default='../api_key')
    parser.add_argument('--freq', help='Get forecasts with this frequence in hours. ', type=int, default=6)
    parser.add_argument('--rate-limit', help='Maximum number of requests per minute', type=int, default=5)
    parser.add_argument('--ref-time-per-request',
                        help='For each request, limit number of reference times to this number',
                        type=int, default=int(1e4))
    parser.add_argument('--overwrite', help='If set, overwrite existing dataset files', action='store_true')

    args = parser.parse_args()

    with open(args.api_key) as fp:
        api_key = fp.readline().strip()

    with open(args.coords_file) as fp:
        csv_reader = csv.DictReader(fp)
        coordinates = [{'longitude': float(row['longitude']), 'latitude': float(row['latitude'])} for row in csv_reader]
    windpower.greenlytics_api.download_coords(args.data_dir, coordinates,
                                              MODEL, args.variables, api_key,
                                              start_date=args.start_date,
                                              end_date=args.end_date,
                                              freq=args.freq,
                                              ref_times_per_request=args.ref_times_per_request,
                                              rate_limit=args.rate_limit,
                                              overwrite=args.overwrite,
                                              )

if __name__ == '__main__':
    main()