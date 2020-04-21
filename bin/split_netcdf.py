import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import datetime
import re

pattern = re.compile(r'(DWD_ICON-EU)_(\d*\.\d*),(\d*\.\d*)_20\d\d-\d\d-\d\d \d\d-20\d\d-\d\d-\d\d \d\d.nc')

def main():
    parser = argparse.ArgumentParser(description="Splits NetCDF files with multiple reference times "
                                                 "into files with single reference times")
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    files = []

    for d in args.directories:
        for f in d.glob('**/*.nc'):
            m = re.match(pattern, f.name)
            if m is not None:
                files.append(f)

    print(files)
    for f in tqdm(files):
        process_dataset(f)
    #with multiprocessing.Pool() as p:
    #    r = list(tqdm(p.imap_unordered(process_dataset, files), total=len(files)))


def process_dataset(f: Path):
    ds = xr.open_dataset(f)
    time_format = '%Y-%m-%d %H'
    m = re.match(pattern, f.name)
    if m is not None:
        model, lat, lon = m.groups()
    else:
        raise ValueError("Could not match filename fields from {}".format(f))
    for i, ref_time in enumerate(ds['reference_time']):
        ds_at_ref_time = ds.isel(reference_time=slice(i, i+1))  # Slice to keep reference time as a dimension
        reference_datetime = datetime.datetime.fromisoformat(str(ds_at_ref_time['reference_time'].values[0]).split('.')[0])
        file_name = f.with_name('{}_{},{}_{}.nc'.format(model, lat, lon, reference_datetime.strftime(time_format)))
        ds_at_ref_time.to_netcdf(file_name)
    ds.close()
    f.unlink()


if __name__ == '__main__':
    main()
