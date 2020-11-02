import argparse
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from windpower.train_ensemble import train



def main():
    parser = argparse.ArgumentParser(description='Train random forrest on sites')
    parser.add_argument('config', help="Python module determining configuration settings", type=Path)
    parser.add_argument('experiment_dir', help="Directory to output results to", type=Path)
    parser.add_argument('split_files_dir', help="Directory with dataset splits to use", type=Path)
    parser.add_argument('site_files', help="NetCDF files to use", nargs='+', type=Path)
    parser.add_argument('--hp-search-iterations', help="Number of hyper parameter search iterations", type=int, default=1)
    args = parser.parse_args()

    command_args = vars(args)
    metadata = dict(command_line_args=command_args)
    splits_files_list = list(args.split_files_dir.glob('site_*_splits.pkl'))
    train(site_files=args.site_files,
          experiment_dir=args.experiment_dir,
          config_path=args.config,
          splits_files_list=splits_files_list,
          metadata=metadata)


if __name__ == '__main__':
    main()



