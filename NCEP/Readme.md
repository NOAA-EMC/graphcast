# GraphCast model with NCEP GDAS Products as ICs

This repository provides scripts to run real-time GraphCast using GDAS products as inputs. There are multiple scripts in the repository including:
- `gdas_utility.py`: a Python script designed to download Global Data Assimilation System (GDAS) data from the National Centers for Environmental Prediction (NCEP) from NOAA S3 bucket (or NOMADS), and prepare the data in a format suitable for feeding into the GraphCast weather prediction system.
- `run_graphcast.py`: a Python script that calls GraphCast and takes GDAS products as input and produces six-hourly forecasts with an arbitrary forecast length (e.g., 40 --> 10-days).
- `graphcast_job_[machine_name].sh`: a Bash script that automates running GraphCast in real-time over the AWS cloud machine (should be submitted through CronJob).

## Table of Contents
- [Overview](#overview)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [Usage](#usage)
  - [GDAS Utility](#gdas-utility)
  - [Run GraphCast](#run-graphcast)
  - [Run GraphCast Through Cronjob](#run-graphcast-through-cronjob)
- [Output](#output)
- [Contact](#contact)

## Overview

The National Centers for Environmental Prediction (NCEP) provides GDAS data that can be used for weather prediction and analysis. This repository simplifies the process of downloading GDAS data, extracting relevant variables, and converting it into a format compatible with the GraphCast weather prediction system. In addition, it automates running GraphCast with GDAS inputs on the NOAA clusters.

## Prerequisites and Installation

To install the package, run the following commands:

```bash
conda create --name mlwp python=3.10
```

```bash
conda activate mlwp
```

```bash
conda install --channel conda-forge pygrib requests bs4
```

```bash
pip install dm-tree boto3 xarray netcdf4
```

```bash
conda install --channel conda-forge cartopy
```

```bash
pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip
```

If you would like to save as grib2 format, the following packages are needed:

```bash
pip install ecmwflibs
````
```bash
pip install iris
````

```bash
pip install iris-grib
````

This will install the packages and most of their dependencies.


Additionally, the utility uses the `wgrib2` library for extracting specific variables from the GDAS data. You can download and install `wgrib2` from [here](http://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/). Make sure it is included in your system's PATH.

## Usage

To use the utility, follow these steps:

Clone the NOAA-EMC GraphCast repository:

```bash   
git clone https://github.com/NOAA-EMC/graphcast.git
```

```bash
cd graphcast/NCEP
```

## GDAS Utility

To download and prepare GDAS data, use the following command:

```bash
python3 gdas_utility.py yyyymmddhh yyyymmddhh --level 13 --source s3 --output /directory/to/output --download /directory/to/download --keep no
```

#### Arguments (required):

- `yyyymmddhh`: Start datetime
- `yyyymmddhh`: End datetime

#### Arguments (optional):

- `-l or --level`: [13, 37], represents the number of pressure levels (default: 13)
- `-s or --source`: [s3, nomads], represents the source to download GDAS data (default: "s3")
- `-o or --output`: /directory/to/output, represents the directory to output netcdf file (default: "current directory")
- `-d or --download`: /directory/to/download, represents the download directory for grib2 files (default: "current directory")
- `-k or --keep`: [yes, no], specifies whether to keep downloaded data after processing (default: "no")

Example usage with options:

```bash
python3 gdas_utility.py 2023060600 2023060606 -o /path/to/output -d /path/to/download
```

Note: 
- The 37 pressure levels option is still under development.
- GraphCast only needs 2 states for initialization, however, gdas_utility can provide longer outputs for evaluation of the model (e.g., 10-days).

   
## Run GraphCast

To run GraphCast, use the following command:

```bash
python3 run_graphcast.py --input /path/to/input/file --output /path/to/output/file --weights /path/to/graphcast/weights --length forecast_length --upload yes --keep no`
```

#### Arguments (required):

- `-i or --input`: /path/to/input/file, represents the path to input netcdf file (including file name and extension)
- `-o or --output`: /path/to/output/file, represents the path to output netcdf file (including file name and extension)
- `-w or --weights`: /path/to/graphcast/weights, represents the path to the parent directory of the graphcast params (weights) and stats from the pre-trained model
- `-l or --length`: An integer number in the range [1, 40], represents the number of forecasts time steps (6-hourly; e.g., 40 → 10-days)

#### Arguments (optional):

- `-u or --upload`: [yes, no], option for uploading the input and output files to NOAA S3 bucket [noaa-nws-graphcastgfs-pds] (default: "no")
- `-k or --keep`: [yes, no], specifies whether to keep input and output files after uploading to NOAA S3 bucket (default: "no")

Example usage with options (1-day forecast):

```bash
python3 run_graphcast.py  -i /path/to/input -o /path/to/output -w /path/to/graphcast/weights -l 4
```

## Run GraphCast Through Cronjob

Submit the `cronjob_[machine_name].sh` to run GraphCast and get real-time (every 6 hours) forecasts through cronjob.

```bash
# Example CronJob to run GraphCast every 6 hours
0 */6 * * * /lustre/Sadegh.Tabas/graphcast/NCEP/cronjob_cloud.sh >> /lustre/Sadegh.Tabas/graphcast/NCEP/logfile.log 2>&1
```

## Output

The processed GDAS data as well as GraphCast forecasts will be saved in NetCDF format in the related directories (uploading to NOAA S3 bucket option is also provided for both input and output files). The files will be named based on the date.

## Contact

For questions or issues, please contact [Sadegh.Tabas@noaa.gov](mailto:Sadegh.Tabas@noaa.gov).
