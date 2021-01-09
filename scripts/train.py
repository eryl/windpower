import argparse
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from windpower.train_ensemble import train
from windpower.dataset import get_site_id


def main():
    parser = argparse.ArgumentParser(description='Train random forrest on sites')
    parser.add_argument('training_config', help="Config file for the training", type=Path)
    parser.add_argument('dataset_dirs', help="The site dataset directories to compare agains", nargs='+',
                        type=Path)
    parser.add_argument('--site-filter', help="Text files constraining files to train on", type=Path)
    parser.add_argument('--split-files-dir', help="Directory with dataset splits to use", type=Path)
    parser.add_argument('--output-dir', help="Where to output experiment data", type=Path, default=Path())
    parser.add_argument('--hp-search-iterations', help="Number of hyper parameter search iterations", type=int, default=1)

    args = parser.parse_args()

    command_args = vars(args)
    metadata = dict(command_line_args=command_args)

    datasets = []
    for dataset_path in args.dataset_dirs:
        if dataset_path.is_dir():
            datasets.extend(dataset_path.glob('**/*.nc'))
        elif dataset_path.suffix == '.nc':
            datasets.append(dataset_path)

    if args.site_filter is not None:
        with open(args.site_filter) as fp:
            site_filter = [l.strip() for l in fp]
            datasets = [dataset_path for dataset_path in datasets if get_site_id(dataset_path) in site_filter]

    splits_files_list = list(args.split_files_dir.glob('site_*_splits.pkl'))
    train(site_files=datasets,
          experiment_dir=args.output_dir,
          config_path=args.training_config,
          splits_files_list=splits_files_list,
          metadata=metadata)


if __name__ == '__main__':
    main()



