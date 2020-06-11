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
    parser.add_argument('--overwrite', help="If set, existing files will be overwritten", action='store_true')
    args = parser.parse_args()

    weather_coordinate_files = defaultdict(dict)
    pattern = re.compile(r'(DWD_ICON-EU|NCEP_GFS)_(\d*\.\d*),(\d*\.\d*).*\.nc')
    for d in args.directories:
        netcdf_files = list(d.glob('*.nc'))
        for f in netcdf_files:
            m = re.match(pattern, f.name)
            if m is not None:
                model, lat, lon = m.groups()
                weather_coordinate_files[(float(lat), float(lon))][model] = f

    for (lat, lon), files in tqdm(weather_coordinate_files.items()):
        try:
            datasets = []
            models = set()
            for model, f in files.items():
                ds = xr.open_dataset(f)
                ds = ds.drop_vars(['phi', 'r', 'latitude', 'longitude'], errors='ignore')
                datasets.append(ds)
                models.add(model.split('_')[0])
            merged_nwp_model = '_'.join(sorted(models))
            #merged_dataset = xr.merge(datasets, join='inner')
            merged_dataset = xr.combine_by_coords(datasets,  join='inner')
            merged_dataset.attrs['nwp_model'] = merged_nwp_model
            args.output_dir.mkdir(exist_ok=True, parents=True)
            output_path = args.output_dir / f"{merged_nwp_model}_{lat},{lon}.nc"
            if output_path.exists() and not args.overwrite:
                print(f"Not overwriting {output_path}, file exists")
            else:
                merged_dataset.to_netcdf(output_path)
            #ds = xr.open_dataset(output_path, engine='netcdf4')
        except ValueError:
            print(f"problematic files: {[f.name for f in files]}")


if __name__ == '__main__':
    main()