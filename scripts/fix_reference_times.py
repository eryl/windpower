import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Fix datetimes")
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    files = []
    for d in args.directories:
        files.extend(d.glob('**/*.nc'))

    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap_unordered(process_dataset, files), total=len(files)))


def process_dataset(f):
    dt_dtype = np.dtype('datetime64[ns]')
    ds = xr.open_dataset(f)
    do_write = False
    if 'reference_time' not in ds.dims:
        ds = ds.expand_dims('reference_time')
        do_write = True
    if ds['reference_time'].dtype != dt_dtype:
        ds['reference_time'] = ds['reference_time'].values.astype(dt_dtype)
        do_write = True
    if do_write:
        ds.to_netcdf(f)

if __name__ == '__main__':
    main()

    