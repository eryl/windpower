import argparse
import re
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

bad_nwp_dates = defaultdict(Counter)

def main():
    parser = argparse.ArgumentParser(description="Filter any reference_times which contains non-finite "
                                                 "values (NaN or infinity).")
    parser.add_argument('weather_datasets', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', help="If set, dataset will be overwritten if it exists", action='store_true')
    args = parser.parse_args()

    datasets = []
    for wds in args.weather_datasets:
        if wds.is_dir():
            datasets.extend(wds.glob('**/*.nc'))
        elif wds.suffix == '.nc':
            datasets.append(wds)

    for ds in tqdm(datasets, desc="Datasets"):
        check_finite(ds, output_dir=args.output_dir, overwrite=args.overwrite)


def check_finite(dataset_path, output_dir=None, overwrite=False):
    ds = xr.open_dataset(dataset_path)
    valid_reference_times = np.full(len(ds['reference_time']), True)
    for name, var in ds.data_vars.items():
        isfinite = np.isfinite(var.values)
        if not np.all(isfinite):
            # Figure out what reference times are bad
            bad_indices = np.argwhere(np.invert(isfinite))
            valid_reference_times[bad_indices[:,0]] = False
    if not np.all(valid_reference_times):
        bad_reference_times = ds['reference_time'].values[np.invert(valid_reference_times)].astype('datetime64[h]')
        print(f"Dataset {dataset_path} has bad values at reference_times {bad_reference_times}")
        ds = ds.sel(reference_time=valid_reference_times)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / dataset_path.name
        else:
            output_path = dataset_path
        if output_path.exists() and not overwrite:
            print(f"Not overwriting {output_path}, file exists and --overwrite is not set")
        else:
            ds.to_netcdf(output_path)


if __name__ == '__main__':
    main()