import xarray as xr
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
#import multiprocessing
import functools
import multiprocessing.dummy as multiprocessing

from windpower.dataset import get_nwp_model_from_path

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
    nwp_model = get_nwp_model_from_path(f)
    do_write = False
    if nwp_model == "DWD_ICON-EU":
        if 'phi' not in ds.variables:
            ds['r'], ds['phi'] = polar_windspeed(ds, 'U', 'V')
            do_write = True
    elif nwp_model == 'FMI_HIRLAM':
        if 'phi' not in ds.variables:
            ds['r'], ds['phi'] = polar_windspeed(ds, 'WindUMS', 'WindVMS')
            do_write = True
    elif nwp_model == 'NCEP_GFS':
        if 'phi' not in ds.variables:
            ds['r'], ds['phi'] = polar_windspeed(ds, 'WindUMS_Height', 'WindVMS_Height')
            do_write = True
    elif nwp_model == 'MetNo_MEPS':
        if 'phi_z' not in ds.variables:
            ds['r_z'], ds['phi_z'] = polar_windspeed(ds, 'x_wind_z', 'y_wind_z')
            ds['r_10m'], ds['phi_10m'] = polar_windspeed(ds, 'x_wind_10m', 'y_wind_10m')
            do_write = True
    elif nwp_model == 'DWD_NCEP':
        if 'dwd_phi' not in ds.variables:
            ds['dwd_r'], ds['dwd_phi'] = polar_windspeed(ds, 'U', 'V')
            do_write = True
        if 'ncep_phi' not in ds.variables:
            ds['ncep_r'], ds['ncep_phi'] = polar_windspeed(ds, 'WindUMS_Height', 'WindVMS_Height')
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

