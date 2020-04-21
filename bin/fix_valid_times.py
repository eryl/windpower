import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing.dummy as multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Make sure the valid_times of all datasets are int32")
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()

    files = []
    for d in args.directories:
        files.extend(d.glob('**/*.nc'))

    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap_unordered(process_dataset, files), total=len(files)))


def process_dataset(f):
    ds = xr.open_dataset(f)
    if ds['valid_time'].dtype != np.int32:
        ds['valid_time'] = ds['valid_time'].astype(np.int32)
        ds.to_netcdf(f)

if __name__ == '__main__':
    main()