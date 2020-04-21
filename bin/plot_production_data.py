import argparse
import pandas as pd
import pytz
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def main():
    parser = argparse.ArgumentParser(description="Plot production data")
    parser.add_argument('metadata_file', help="File containing the metadata about sites")
    parser.add_argument('output_dir', help="Where to save files", type=Path)
    parser.add_argument('production_files', help="Files to load production data from", nargs='+')
    args = parser.parse_args()

    metadata = pd.read_excel(args.metadata_file).dropna(thresh=15)  # The metadata file is 16 columns, one have some missing vnalues which doesn't matter

    site_metadata = dict()
    lats = []
    lons = []
    for i in range(len(metadata)):
        row = dict(metadata.iloc[i].items())
        lats.append(row['Lat'])
        lons.append(row['Lon'])

    production_dfs = [pd.read_csv(m, sep=',', parse_dates=True, index_col=0, header=0, skiprows=[1]) for m in
                       args.production_files]
    data = pd.concat(production_dfs, axis=1)
    capacities_dfs = [pd.read_csv(m, sep=',', parse_dates=True, index_col=0, header=0, nrows=1) for m in
                       args.production_files]
    capacities = pd.concat(capacities_dfs, axis=1)
    #date_index = data.index

    #tz_stockholm = pytz.timezone('Europe/Stockholm')
    #localized_dates = [tz_stockholm.localize(d) for d in date_index]
    #utc_dts = [dt.astimezone(pytz.utc) for dt in localized_dates]
    #data.set_index(np.array(utc_dts))
    for site_id in data.columns:
        site_production = data[site_id].dropna()
        capacity = capacities[site_id].to_numpy()
        normalized_production = site_production / capacity
        fig = plt.figure(figsize=(16,8))
        normalized_production.plot(ax=plt.gca())
        n_hours = len(normalized_production)
        zero_production_intervals = zero_runs(normalized_production.to_numpy())
        interval_lengths = zero_production_intervals[:, 1] - zero_production_intervals[:, 0]
        zero_production_index = 1 - np.sqrt(np.sum(interval_lengths**2)) / n_hours
        fig.savefig(args.output_dir / '{:.4%}_{}_{}.png'.format(zero_production_index, n_hours, site_id))
        plt.close()


if __name__ == '__main__':
    main()