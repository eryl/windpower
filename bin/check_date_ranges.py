import datetime

import xarray as xr
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Merge files")
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()

    weather_coordinate_files = defaultdict(list)
    coord_fmt = r"\d+\.\d+"
    model_fmt = r"DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS"
    date_fmt = r"\d\d\d\d-\d\d-\d\d \d\d"
    date_pattern = r"({})_({}),({})_({})--({}).nc".format(model_fmt, coord_fmt, coord_fmt, date_fmt, date_fmt)
    nondate_pattern = r"({})_({}),({}).nc".format(model_fmt, coord_fmt, coord_fmt)

    dataset_files = []
    for f in args.files:
        if f.is_dir():
            dataset_files.extend(f.glob('**/*.nc'))
        else:
            dataset_files.append(f)

    for f in dataset_files:
        m = re.match(date_pattern, f.name)
        if m is not None:
            model, latitude, longitude, start_date, end_date = m.groups()
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H')
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H')
            ds = xr.open_dataset(f)
            reference_time = ds['reference_time'].values
            min_date = datetime.datetime.utcfromtimestamp(min(reference_time).tolist() / 1e9)
            max_date = datetime.datetime.utcfromtimestamp(max(reference_time).tolist() / 1e9)
            if start_date != min_date or end_date != max_date:
                print(f"Incorrect ranges for {f}, filename: {start_date}--{end_date}, in dataset: {min_date}--{end_date}")

        else:
            raise ValueError(f"Not a valid NWP dataset file name: {f}")



if __name__ == '__main__':
    main()