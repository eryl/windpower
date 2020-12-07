import argparse
import itertools
from collections import defaultdict
from pathlib import Path

from mltrain.util import load_config

from windpower.dataset import get_site_id, SplitConfig, make_site_splits, get_nwp_model
from windpower.train_ensemble import train

def main():
    parser = argparse.ArgumentParser("Perform pairwise trainings on the given site dataset folders")
    parser.add_argument('training_config', help="Config file for the training", type=Path)
    parser.add_argument('dataset_dirs', help="The site dataset directories to compare agains", nargs='+',
                        type=Path)
    parser.add_argument('--site-filter', help="If given, should be a file with site id's to train on", type=Path)
    parser.add_argument('--output-dir', help="Where to output experiment data", type=Path, default=Path())
    parser.add_argument('--exclude-pairs', help="Exclude these comma-seperated pairs", nargs='+')

    args = parser.parse_args()

    split_config = load_config(args.training_config, SplitConfig)

    exclude_pairs = set()
    if args.exclude_pairs is not None:
        for pair in args.exclude_pairs:
            a, b = pair.split(',')
            exclude_pairs.add(frozenset((a, b)))

    datasets = []
    for dataset_path in args.dataset_dirs:
        if dataset_path.is_dir():
            datasets.extend(dataset_path.glob('**/*.nc'))
        elif dataset_path.suffix == '.nc':
            datasets.append(dataset_path)

    site_datasets = defaultdict(set)
    for dataset_path in datasets:
        site_id = get_site_id(dataset_path)
        site_datasets[site_id].add(dataset_path)

    if args.site_filter is not None:
        with open(args.site_filter) as fp:
            site_filter = [l.strip() for l in fp]
            site_datasets = {site_id: dataset_paths for site_id, dataset_paths in site_datasets.items() if site_id in site_filter}

    for site_id, site_paths in site_datasets.items():
        if len(site_paths) < 2:
            print(f"Warning, site {site_id} does not have enough datasets to perform pairwise tests")
        for site_a, site_b in itertools.combinations(site_paths, 2):
            site_list = [site_a, site_b]
            site_a_model = get_nwp_model(site_a)
            site_b_model = get_nwp_model(site_b)

            if frozenset((site_a_model.identifier, site_b_model.identifier)) in exclude_pairs:
                print(f"Excluding pair ({site_a_model.identifier}, {site_b_model.identifier})")
                continue
            if site_a_model.identifier == site_b_model.identifier:
                print(f"Excluding {site_a}, {site_b}, they are the same NWP model")
                continue
            output_dir = args.output_dir / f'{site_a_model.identifier}-vs-{site_b_model.identifier}'
            splits_file = make_site_splits(site_id, list(site_paths), output_dir, split_config)
            train(site_files=site_list,
                  splits_files_list=[splits_file],
                  experiment_dir=output_dir,
                  config_path=args.training_config)


if __name__ == '__main__':
    main()