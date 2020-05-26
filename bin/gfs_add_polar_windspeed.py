import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing


def main():
    parser = argparse.ArgumentParser(description="Add polar windspeeds for DWD datasets")
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    files = []
    for d in args.directories:
        files.extend(d.glob('**/*.nc'))

    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap_unordered(process_dataset, files), total=len(files)))


def process_dataset(f):
    ds = xr.open_dataset(f)

    if 'phi' not in ds.variables:
        x = ds['U']  # Zonal wind component
        y = ds['V']  # Meridional wind component
        ds['r'] = np.sqrt(x**2 + y**2)
        ds['phi'] = np.arctan2(y, x)
        ds.to_netcdf(f)

if __name__ == '__main__':
    main()

