import argparse
from pathlib import Path
import re

def main():
    parser = argparse.ArgumentParser(description="Find the site LPs for the plots in a directory")
    parser.add_argument('plot_directory', help="Directory containing the plots", type=Path)
    parser.add_argument('output', help="Directory containing the plots", type=argparse.FileType('w'))
    args = parser.parse_args()
    plot_files = args.plot_directory.glob('*.png')
    site_pattern = re.compile(r'\d+\.\d+%_\d+_(\d+).png')
    site_lps = []
    for plot in plot_files:
        match = site_pattern.match(plot.name)
        if match is not None:
            site_id = match.group(1)
            site_lps.append(site_id)
    args.output.write('\n'.join(sorted(site_lps)))

if __name__ == '__main__':
    main()



