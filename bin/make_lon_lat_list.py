import argparse
from pathlib import Path
import pandas as pd
import csv

def main():
    parser = argparse.ArgumentParser(description="Make a list of longitude and latitude pairs for the selected sites")
    parser.add_argument('metadata_file', help='CSV File containing the metadata', type=Path)
    parser.add_argument('sites_file', help='File containing the LPs of sites to write down pairs of', type=Path)
    parser.add_argument('lon_lat_file', help='File to write longitude and latitude pairs to.', type=Path)

    args = parser.parse_args()

    metadata = pd.read_excel(args.metadata_file).dropna(thresh=15)  # The metadata file is 16 columns, one have some missing vnalues which doesn't matter
    with open(args.sites_file) as fp:
        sites = [site.strip() for site in fp if site.strip()]

    coordinates = set()
    for site in sites:
        site_row = metadata[metadata['LP'] == site]
        lat = float(site_row['Lat'])
        lon = float(site_row['Lon'])
        coordinates.add((lat, lon))

    with open(args.lon_lat_file, 'w') as fp:
        csv_writer = csv.DictWriter(fp, fieldnames=['longitude', 'latitude'])
        csv_writer.writeheader()
        for lat, lon in sorted(coordinates):
            csv_writer.writerow(dict(longitude=lon, latitude=lat))

if __name__ == '__main__':
    main()
