import argparse
import itertools
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from windpower.mltrain.util import load_config

from windpower.dataset import get_site_id, SplitConfig, make_site_splits, get_nwp_model
from windpower.train_ensemble import train

def main():
    parser = argparse.ArgumentParser("Create dataset splits")
    parser.add_argument('training_config', help="Config file for the training", type=Path)
    parser.add_argument('dataset_dirs', help="The site dataset directories to make splits for agains", nargs='+',
                        type=Path)
    parser.add_argument('--output-dir', help="Where to output the splits files", type=Path, default=Path())

    args = parser.parse_args()

    split_config = load_config(args.training_config, SplitConfig)

    datasets = []
    for dataset_path in args.dataset_dirs:
        if dataset_path.is_dir():
            datasets.extend(dataset_path.glob('**/*.nc'))
        elif dataset_path.suffix == '.nc':
            datasets.append(dataset_path)

    for dataset_path in tqdm(datasets, desc="Dataset"):
        site_id = get_site_id(dataset_path)
        site_model = get_nwp_model(dataset_path)
        output_dir = args.output_dir / f'{site_model.identifier}'
        splits_file = make_site_splits(site_id, [dataset_path], output_dir, split_config)



if __name__ == '__main__':
    main()