import functools

import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing
#import multiprocessing.dummy as multiprocessing

from windpower.dataset import get_nwp_model_from_path

def main():
    parser = argparse.ArgumentParser(description="Fix datetimes")
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output-dir', help="Directory to write results to", type=Path)
    parser.add_argument('--overwrite', help="If this flag is set, any existing file will be overwritten",
                        action='store_true')
    parser.add_argument('--from-filename',
                        help="If this is set, the nwp_model attribute will always be derived from the filename, "
                             "even if a 'nwp_model' data array is present. This also overwrites any existing such "
                             "attribute",
                        action='store_true')
    args = parser.parse_args()
    files = []
    for d in args.directories:
        files.extend(d.glob('**/*.nc'))

    f = functools.partial(process_dataset, output_dir=args.output_dir, overwrite=args.overwrite,
                          from_filename=args.from_filename)
    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap_unordered(f, files), total=len(files)))


def process_dataset(f, output_dir=None, overwrite=False, from_filename=False):
    ds = xr.open_dataset(f)
    do_write = False
    if 'nwp_model' in ds.data_vars:
        nwp_model = str(ds['nwp_model'].values)
        ds = ds.drop_vars('nwp_model')
        if not from_filename:
            ds.attrs['nwp_model'] = nwp_model
        do_write = True
    if from_filename or 'nwp_model' not in ds.attrs:
        nwp_model = get_nwp_model_from_path(f)
        ds.attrs['nwp_model'] = nwp_model
        do_write = True
    if do_write:
        print(f"Fixed attributes of {f}")
        if output_dir is not None:
            output_path = output_dir / f.name
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = f
        if output_path.exists() and not overwrite:
            print(f"Not overwriting {f}, file exists and --overwrite is not set")
        else:
            ds.to_netcdf(output_path)


if __name__ == '__main__':
    main()

    