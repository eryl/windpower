import xarray as xr
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import re
from collections import defaultdict, Counter
from windpower.dataset import SiteDataset, DEFAULT_VARIABLE_CONFIG, DEFAULT_DATASET_CONFIG
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Merge files")
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()

    dataset_files = []
    for d in args.directories:
        if d.is_dir():
            dataset_files.extend(d.glob('*.nc'))
        else:
            dataset_files.append(d)

    dataset_sizes = defaultdict(list)
    min_size = None
    max_size = None

    for f in tqdm(dataset_files, desc="Dataset files processed"):
        dataset = SiteDataset(dataset_path=f,
                              dataset_config=DEFAULT_DATASET_CONFIG,
                              variables_config=DEFAULT_VARIABLE_CONFIG)
        nwp_model =dataset.get_nwp_model()
        size = len(dataset)
        if min_size is None or size < min_size:
            min_size = size
        if max_size is None or size > max_size:
            max_size=size
        dataset_sizes[nwp_model].append(size)

    #fig, axes = plt.subplots(len(dataset_sizes), 1)
    s_models = []
    s_sizes = []
    for nwp_model, sizes in dataset_sizes.items():
        s_sizes.extend(sizes)
        s_models.extend([nwp_model]*len(sizes))

    counted_sizes = {nwp_model: Counter(sizes) for nwp_model, sizes in dataset_sizes.items()}
    print(counted_sizes)
    #dataset_sizes = pd.DataFrame(dict(sizes=s_sizes, models=s_models))
    #for (nwp_model, sizes), ax in zip(sorted(dataset_sizes.items()), axes.flatten()):

        #sns.distplot(sizes, hist_kws=dict(range=(min_size, max_size)), label=nwp_model, kde=False)
        #sns.distplot(sizes, bins=100, label=nwp_model, kde=False, ax=ax)
        #ax.legend()
    #sns.swarmplot(x='models', y='sizes', data=dataset_sizes)
    #plt.show()

if __name__ == '__main__':
    main()