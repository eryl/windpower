import argparse
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from windpower.train_ensemble import train



def main():
    parser = argparse.ArgumentParser(description='Train random forrest on sites')
    parser.add_argument('variables_config', help="Python module determining what variables to use, "
                                               "and how they should be encoded", type=Path)
    parser.add_argument('model_config',
                        help="Python module determining what model and hyper parameters to use",
                        type=Path)
    parser.add_argument('dataset_config',
                        help="Python module determining dataset parameters to use",
                        type=Path)
    parser.add_argument('training_config',
                        help="Python module determining training parameters to use",
                        type=Path)
    parser.add_argument('experiment_dir', help="Directory to output results to", type=Path)

    parser.add_argument('site_files', help="NetCDF files to use", nargs='+', type=Path)
    args = parser.parse_args()

    train(site_files=args.site_files,
          experiment_dir=args.experiment_dir,
          training_config_path=args.training_config,
          dataset_config_path=args.dataset_config,
          model_config_path=args.model_config,
          variables_config_path=args.variables_config)


if __name__ == '__main__':
    main()



