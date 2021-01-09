import argparse
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from windpower.train_ensemble import evaluate_model



def main():
    parser = argparse.ArgumentParser(description='Test a trained model on specified site data')
    parser.add_argument('model_path', help="Path to model", type=Path)
    parser.add_argument('dataset_path', help="Path to site dataset", type=Path)
    parser.add_argument('output_dir', help="Path to output results to", type=Path)
    parser.add_argument('--split-file', help="Path to file containing date time splits", type=Path)
    parser.add_argument('--split-label', help="If split file is supplied, what label to use.", default='test')
    args = parser.parse_args()

    command_args = vars(args)
    if args.split_file is not None:
        reference_times = np.load(args.split_file)[args.split_label]
    else:
        reference_times = None
    evaluate_model(reference_times, args.model_path, args.output_dir, args.dataset_path)


if __name__ == '__main__':
    main()

