# Wind Power Forecasting Test Bed
Python package for performing wind power forecast based on numerical weather forecasts.

The package is based on the idea that there is some target you want to predict based on NWPs and is
structured around handling this problem easily. It is assumed that the target is spatial (i.e. has a geographic position).
The intended use is for wind power prediction but could be used to predict any variable based on spatial data (e.g. ice
cream consumption or solar power production)

# Target
The target variable is defined by its geographical position and measurements.
Location information should be a in a CSV file with three columns, the location id, latitude and longitude.
Measurments should be a csv file where the first column is the time, and the remaining columns are production values for
the locations over time.

# Downloading data
Data is downloaded based on the location csv's. The coordinate file determines which coordinates are downloaded while
the measurements CSV file determines what date intervals will be downloaded.

# Configuration
All configuration is based on python files.
There are three main configurations:
  - Dataset layout: This determines who the forecasts are converted into dataset, how they are windowed, what
    measurement to predict and so on
  - Variable definition: This defines what variables from the NWP should be used and how they should be encoded.
  - Training config: This defines how the training should be performed, such as cross validation strategy
