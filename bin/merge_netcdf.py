import xarray as xr
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge files")
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()


    for d in args.directories:
        files = list(d.glob('*.nc'))


