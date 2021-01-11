import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser("Merge the input csv files")
parser.add_argument('csv_files', help="files to merge", type=Path, nargs='+')
parser.add_argument('--output-file', help="Where to write output, to standard out if not given", type=Path)
args = parser.parse_args()

merged_data = pd.concat([pd.read_csv(csv_file) for csv_file in args.csv_files])
if args.output_file is not None:
    print(f"Writing data to {args.output_file}")
    merged_data.to_csv(args.output_file, index=False)
else:
    print(merged_data)
