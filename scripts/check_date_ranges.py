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
    parser.add_argument('--diff', help="Any difference in days between filename and reference time dates greater "
                                       "than this will be reported", type=int, default=3)
    args = parser.parse_args()

    dt = datetime.timedelta(days=args.diff)

    coord_fmt = r"\d+\.\d+"
    model_fmt = r"DWD_ICON-EU|FMI_HIRLAM|NCEP_GFS|MEPS|MetNo_MEPS"
    date_fmt = r"\d\d\d\d-\d\d-\d\d \d\d"
    date_pattern = r"({})_({}),({})_({})--({}).nc".format(model_fmt, coord_fmt, coord_fmt, date_fmt, date_fmt)

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
            min_date = min(reference_time).astype('datetime64[s]').tolist()
            max_date = max(reference_time).astype('datetime64[s]').tolist()

            if abs(start_date - min_date) > dt or abs(end_date - max_date) > dt:
                print(f"Incorrect ranges for {f}"
                      f"\n\tin name:    {start_date} -- {end_date}"
                      f"\n\tin dataset: {min_date} -- {max_date}")
        else:
            raise ValueError(f"Not a valid NWP dataset file name: {f}")



if __name__ == '__main__':
    main()