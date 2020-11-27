import xarray as xr
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import re
from collections import defaultdict
import windpower.greenlytics_api
from windpower.greenlytics_api import MODEL_MAP, prefix_merge_variable, merge_models, GreenlyticsModelDataset

def main():
    parser = argparse.ArgumentParser(description="Merge files")
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--overwrite', help="If set, existing files will be overwritten", action='store_true')
    args = parser.parse_args()

    weather_coordinate_files = defaultdict(dict)

    for d in args.directories:
        netcdf_files = list(d.glob('*.nc'))
        for f in netcdf_files:
            try:
                model_metadata = GreenlyticsModelDataset.fromstring(f.name)
                weather_coordinate_files[(model_metadata.latitude, model_metadata.longitude)][model_metadata.model.identifier] = f
            except ValueError:
                print(f"Could not parse filename {f}")


    for (lat, lon), files in tqdm(weather_coordinate_files.items()):
        try:
            datasets = []
            models = []

            for model_indentifier, f in files.items():
                ds = xr.open_dataset(f)
                model = MODEL_MAP[model_indentifier]
                valid_variables = set(model.variables + ['height', 'reference_time', 'valid_time'])
                drop_vars = set(ds.variables) - valid_variables
                ds = ds.drop_vars(list(drop_vars), errors='ignore')
                variables_to_rename = set(model.variables).intersection(set(ds.variables.keys()))
                ds = ds.rename({variable: prefix_merge_variable(model_indentifier, variable) for variable in variables_to_rename})
                datasets.append(ds)
                models.append(model)
            merged_nwp_model = merge_models(models)
            #merged_dataset = xr.merge(datasets, join='inner')

            merged_dataset = xr.combine_by_coords(datasets,  join='inner')
            merged_dataset.attrs['nwp_model'] = merged_nwp_model.identifier
            args.output_dir.mkdir(exist_ok=True, parents=True)
            output_path = args.output_dir / f"{merged_nwp_model.identifier}_{lat},{lon}.nc"
            if output_path.exists() and not args.overwrite:
                print(f"Not overwriting {output_path}, file exists")
            else:
                merged_dataset.to_netcdf(output_path)
            #ds = xr.open_dataset(output_path, engine='netcdf4')
        except ValueError:
            print(f"problematic files: {[f.name for identifier, f in files.items()]}")
            raise


if __name__ == '__main__':
    main()