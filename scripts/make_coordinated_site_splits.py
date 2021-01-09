import argparse
from pathlib import Path
from windpower.dataset import SplitConfig, make_all_site_splits
from windpower.mltrain.util import load_config

def main():
    parser = argparse.ArgumentParser(description='Script to generate datetime splits')
    parser.add_argument('dataset_dirs', help="Directory to scan for site datasets", type=Path, nargs='+')
    parser.add_argument('--output-dir', help="Output split files to this directory", type=Path, default=Path('.'))
    parser.add_argument('--split-config', help="File to use for getting split configurations")
    args = parser.parse_args()

    split_config = load_config(args.split_config, SplitConfig)
    datasets = []

    for dataset_path in args.dataset_dirs:
        if dataset_path.is_dir():
            datasets.extend(dataset_path.glob('**/*.nc'))
        elif dataset_path.suffix == '.nc':
            datasets.append(dataset_path)

    make_all_site_splits(datasets, args.output_dir, split_config)



if __name__ == '__main__':
    main()
