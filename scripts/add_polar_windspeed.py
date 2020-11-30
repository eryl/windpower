import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing
#import multiprocessing.dummy as multiprocessing
import functools
from collections import defaultdict


from windpower.greenlytics_api import WIND_SPEED_PAIRS, prefix_split_variable, prefix_merge_variable

def main():
    parser = argparse.ArgumentParser(description="Add polar windspeeds for datasets")
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    files = []
    for d in args.directories:
        files.extend(d.glob('**/*.nc'))

    f = functools.partial(process_dataset, output_dir=args.output_dir, overwrite=args.overwrite)
    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap_unordered(f, files), total=len(files)))


def process_dataset(f, output_dir=None, overwrite=False):
    ds = xr.open_dataset(f)
    do_write = False
    variables_and_prefix = defaultdict(set)
    for variable in ds.variables:
        parts = prefix_split_variable(variable)
        try:
            prefix, var = parts
        except ValueError:
            prefix = None
            var = parts[0]
        variables_and_prefix[prefix].add(var)

    for u_name, v_name in WIND_SPEED_PAIRS:
        for prefix, vars in variables_and_prefix.items():
            if u_name in vars and v_name in vars:
                prefixed_u_name = prefix_merge_variable(prefix, u_name)
                prefixed_v_name = prefix_merge_variable(prefix, v_name)
                name = f'{u_name}_{v_name}'
                r, phi = polar_windspeed(ds, prefixed_u_name, prefixed_v_name)
                ds[prefix_merge_variable(prefix, f'r_{name}')] = r
                ds[prefix_merge_variable(prefix, f'phi_{name}')] = phi
                do_write = True

    if do_write:
        if output_dir is not None:
            output_path = output_dir / f.name
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = f
        if output_path.exists() and not overwrite:
            print(f"Not overwriting {f}, file exists and --overwrite is not set")
        else:
            ds.to_netcdf(output_path)


def polar_windspeed(ds, u_key, v_key):
    x = ds[u_key]
    y = ds[v_key]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(phi))):
        raise ValueError("Polar coordinates are nonfinite")
    return r, phi


if __name__ == '__main__':
    main()

