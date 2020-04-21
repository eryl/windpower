import argparse
import numpy as np
import pandas as pd
import pytz
import pathlib

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

dwd_icon_params = {
    'model': 'DWD_ICON-EU',
    'start_date': '2019-08-15 00',
    'end_date': '2019-08-20 00',
    'coords': {'latitude': [50], 'longitude': [50], 'height': [54, 55, 58, 59, 60, 61]},
    'variables': ['T', 'U', 'V']
}



def main():
    parser = argparse.ArgumentParser(description='Script for downloading NWP data from the Greenlytics API')
    parser.add_argument('metadata_file', help='CSV File containing the metadata')
    parser.add_argument('data_dir', help='Directory to store data to')
    parser.add_argument('--variables', help='The variables to download', nargs='+', required=True)
    parser.add_argument('--n-processes', help='How many processes to use for downloading', type=int, default=1)

    args = parser.parse_args()

    metadata = pd.read_excel(args.metadata_file).dropna(thresh=15)  # The metadata file is 16 columns, one have some missing vnalues which doesn't matter

    site_metadata = dict()
    lats = []
    lons = []
    for i in range(len(metadata)):
        row = dict(metadata.iloc[i].items())
        lats.append(row['Lat'])
        lons.append(row['Lon'])

    measueremnt_dfs = [pd.read_csv(m, sep=';', parse_dates=True, index_col=0, header=0, skiprows=[1]) for m in
                       args.measurements]

    data = pd.concat(measueremnt_dfs, axis=1, verify_integrity=True)
    date_index = data.index

    tz_stockholm = pytz.timezone('Europe/Stockholm')
    localized_dates = [tz_stockholm.localize(d.to_pydatetime()) for d in date_index]
    utc_dts = [dt.astimezone(pytz.utc) for dt in localized_dates]





if __name__ == '__main__':
    main()