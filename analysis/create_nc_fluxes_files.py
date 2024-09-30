#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from typing import Tuple
import xarray as xr

from script_variables import *


######################## READ ME ########################
# This script is no longer necessary as /hybrid/src/postprocess.py has been updated to include the functionality of this script.
# This script is used to create netCDF files from the Fortran binary files output by SPEEDY and HYBRID.
# Ensure postprocess.py is executed after running the model. See `robust_runs.sh` for an example.
#########################################################


def read_flx(filename: str) -> np.ndarray:
    """Read a SPEEDY Fortran binary file and return a numpy array.

    Args:
        filename (str): The name of the file to read.

    Returns:
        np.ndarray: The data from the file.
    """
    f = np.fromfile(filename, dtype=np.float64)
    # shape = (nlon, nlat, nrec)
    shape = (nlon, nlat, -1)
    data = np.reshape(f, shape, order="F")
    # data = data.astype(np.float64)
    return data


def read_files_2_nc(folder: str, n_files: int) -> Tuple[xr.Dataset]:
    """Read the Fortran binary files and return a Dataset with the data.

    Args:
        folder (str): The folder containing the Fortran binary files.
        n_files (int): The number of files to read.

    Returns:
        Tuple[xr.Dataset]: The Datasets containing the data.
    """
    # Loop through all the Fortran binary files
    ds_cc = xr.Dataset()
    ds_tsr = xr.Dataset()
    ds_olr = xr.Dataset()

    cloudc = []
    tsr = []
    olr = []


    # for date in filenames:              
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file and if correct date
        if os.path.isfile(f) and "grd" in f and "fluxes.grd" in f:
            # Create a DataArray for each level along the 'z' dimension
            fdata = read_flx(f)
            cloudc.append(fdata[:, :, 0])
            tsr.append(fdata[:, :, 4])
            olr.append(fdata[:, :, 5])

    var_name = f'cloudc'
    ds_cc = xr.DataArray(np.stack(cloudc, axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
    ds_cc[var_name] = ds_cc

    var_name = f'tsr'
    ds_tsr = xr.DataArray(np.stack(tsr, axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
    ds_tsr[var_name] = ds_tsr

    var_name = f'olr'
    ds_olr = xr.DataArray(np.stack(olr, axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
    ds_olr[var_name] = ds_olr
    
    return ds_cc, ds_tsr, ds_olr



#######################################################
# Find the file names

# Dimensions of the grid
nlon = 96
nlat = 48
nlev = 8
nrec = 10

n_files = (3652*4)
print(n_files)
output_dir = os.path.join(analysis_root, 'annual')


if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

#################### Hybrid ####################
if HYBRID:
    print("Start HYBRID")
    # Set the directory where the Fortran binary files are located
    ds_cc, ds_tsr, ds_olr = read_files_2_nc(os.path.join(HYBRID_data_root, GP_name), n_files)
    # Write the Dataset to a netCDF4 file
    ds_cc.to_netcdf(os.path.join(output_dir,'HYBRID_cloudc.nc'), engine='netcdf4')
    ds_tsr.to_netcdf(os.path.join(output_dir,'HYBRID_tsr.nc'), engine='netcdf4')
    ds_olr.to_netcdf(os.path.join(output_dir,'HYBRID_olr.nc'), engine='netcdf4')
#################### SPEEDY ####################
if SPEEDY:
    print("Start SPEEDY")
    # Set the directory where the Fortran binary files are located
    ds_cc, ds_tsr, ds_olr = read_files_2_nc(os.path.join(SPEEDY_root), n_files)
    # Write the Dataset to a netCDF4 file
    ds_cc.to_netcdf(os.path.join(output_dir,'SPEEDY_cloudc.nc'), engine='netcdf4')
    ds_tsr.to_netcdf(os.path.join(output_dir,'SPEEDY_tsr.nc'), engine='netcdf4')
    ds_olr.to_netcdf(os.path.join(output_dir,'SPEEDY_olr.nc'), engine='netcdf4')

