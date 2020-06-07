import argparse
import random
import re
from collections import defaultdict, Counter
import shutil
from pathlib import Path


bad_nwp_dates = defaultdict(Counter)

def main():
    parser = argparse.ArgumentParser(description="Select a subset of the sites in the given directories.")
    parser.add_argument('site_datasets', type=Path, nargs='+')
    parser.add_argument('destination_dir', type=Path)
    parser.add_argument('-n', help="Number of sites to keep in the subset. This and 'ratio' are mutally exclusive.",
                        type=int)
    parser.add_argument('--ratio', help="Ratio of sites to keep in the subset. This and '--n' are mutually exclusive.",
                        type=float)
    parser.add_argument('--full-set-only',
                        help="If this flag is set, only sample from sites which have the maximum "
                             "number of site datasets (excluding sites with only "
                             "some of the nwp dataset)",
                        action='store_true')
    args = parser.parse_args()

    if args.n is not None and args.ratio is not None:
        raise ValueError("Only one of '-n' and '--ratio' can be set.")

    site_files = []
    for f in args.site_datasets:
        if f.is_dir():
            site_files.extend(f.glob('**/*.nc'))
        else:
            site_files.append(f)

    site_ids_to_files = defaultdict(list)
    for f in site_files:
        pattern = r"(\d+)_[\w_-]+.nc"
        m = re.match(pattern, f.name)
        if m is not None:
            site_id, = m.groups()
            site_ids_to_files[int(site_id)].append(f)

    if args.full_set_only:
        n_before_filter = len(site_ids_to_files)
        full_set_n = max(len(v) for v in site_ids_to_files.values())
        site_ids_to_files = {k:v for k,v in site_ids_to_files.items() if len(v) == full_set_n}
        print(f"After filtering only full set sites, {len(site_ids_to_files)}/{n_before_filter} sites remain")

    if args.ratio is not None:
        n = int(len(site_ids_to_files)*args.ratio)
    else:
        n = args.n

    if n >= len(site_ids_to_files):
        raise ValueError(f"Can't produce a subsample of size {n} from selected sites")

    subsample = random.sample(site_ids_to_files.keys(), n)
    for selected in subsample:
        files = site_ids_to_files[selected]
        for f in files:
            shutil.copy(f, args.destination_dir)



if __name__ == '__main__':
    main()

