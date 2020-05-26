import xarray as xr
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Merge files")
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    weather_coordinate_files = defaultdict(list)
    pattern = re.compile(r'(DWD_ICON-EU|NCEP_GFS)_(\d*\.\d*),(\d*\.\d*)_.*\.nc')
    for d in args.directories:
        for f in d.glob('*.nc'):
            m = re.match(pattern, f.name)
            if m is not None:
                model, lat, lon = m.groups()
                weather_coordinate_files[(model, float(lat), float(lon))].append(f)
    for (model, lat, lon), files in tqdm(weather_coordinate_files.items()):
        ds = xr.open_mfdataset(sorted(files))
        unique_reference_times, indices = np.unique(ds['reference_time'].values, return_index=True)
        filtered_ds = ds.isel(reference_time=indices)
        args.output_dir.mkdir(exist_ok=True, parents=True)
        filtered_ds.to_netcdf(args.output_dir / f"{model}_{lat},{lon}.nc")


if __name__ == '__main__':
    main()