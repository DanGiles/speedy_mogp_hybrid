#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Tuple
import numpy as np
from datetime import datetime, timedelta
import xarray as xr

from script_variables import *


######################## READ ME ########################
# This script is no longer necessary as /hybrid/src/postprocess.py has been updated to include the functionality of this script.
# This script is used to create netCDF files from the Fortran binary files output by SPEEDY and HYBRID.
# Ensure postprocess.py is executed after running the model. See `robust_runs.sh` for an example.
#########################################################


def read_grd(filename: str) -> np.ndarray:
    """Read a SPEEDY Fortran binary file and return a numpy array.

    Args:
        filename (str): The name of the file to read.

    Returns:
        np.ndarray: The data from the file.
    """
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data


def read_files_2_precip_nc(folder: str, n_files: int) -> xr.Dataset:
    """Read the Fortran binary files and return a Dataset with the precipitation data.
    
    Args:
        folder (str): The folder containing the Fortran binary files.
        n_files (int): The number of files to read.
        
    Returns:
        xr.Dataset: The Dataset containing the precipitation data.
    """
    # Loop through all the Fortran binary files

    ds_precip = xr.Dataset()
    precip = []
    # for date in filenames:              
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file and if correct date
        if os.path.isfile(f) and "grd" in f and "fluxes.grd" not in f:
            # Create a DataArray for each level along the 'z' dimension
            fdata = read_grd(f)
            precip.append(fdata[:, :, 33])

    var_name = f'precip'
    da_precip = xr.DataArray(np.stack(precip, axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
    ds_precip[var_name] = da_precip
    
    return ds_precip


def read_files_2_nc(folder: str, n_files: int) -> Tuple[xr.Dataset]:
    """Read the Fortran binary files and return a Dataset with the temperature, specific humidity, surface pressure and precipitation data.

    Args:
        folder (str): The folder containing the Fortran binary files.
        n_files (int): The number of files to read.

    Returns:
        Tuple[xr.Dataset]: The Datasets containing the temperature, specific humidity, surface pressure and precipitation data.
    """
    # Loop through all the Fortran binary files
    ds_t = xr.Dataset()
    ds_q = xr.Dataset()
    ds_ps = xr.Dataset()
    ds_precip = xr.Dataset()

    ps = []
    precip = []
    T = [[] for _ in range(8)]
    Q = [[] for _ in range(8)]

    # for date in filenames:              
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file and if correct date
        if os.path.isfile(f) and "grd" in f and "fluxes.grd" not in f:
        # if os.path.isfile(f) and date in f and "_fluxes.grd" not in f:
            # Create a DataArray for each level along the 'z' dimension
            fdata = read_grd(f)
            ps.append(fdata[:, :, 32])
            precip.append(fdata[:, :, 33])
            for z in range(8):
                T[z].append(fdata[:, :, z+16])
            for z in range(8):
                Q[z].append(fdata[:, :, z+24])

    # Create a DataArray for each level along 'z' and assign it to the Dataset
    for z in range(8):
        var_name = f'T_{z}'
        da_t = xr.DataArray(np.stack(T[z], axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
        ds_t[var_name] = da_t

    for z in range(8):
        var_name = f'Q_{z}'
        da_q = xr.DataArray(np.stack(Q[z], axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
        ds_q[var_name] = da_q

    var_name = f'ps'
    da_ps = xr.DataArray(np.stack(ps, axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
    ds_ps[var_name] = da_ps

    var_name = f'precip'
    da_precip = xr.DataArray(np.stack(precip, axis=0), coords={'timestamp': range(n_files), 'lon': range(96), 'lat': range(48)}, dims=['timestamp', 'lon', 'lat'], name=var_name)
    ds_precip[var_name] = da_precip
    
    return ds_t, ds_q, ds_ps, ds_precip



#######################################################
# Find the file names

# Dimensions of the grid
nlon = 96
nlat = 48
nlev = 8
nrec = 10

# Prepare the December, January and February data
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Time increments
delta = timedelta(hours=6)

# Initial Date
idate = "1982010100"
SPEEDY_DATE_FORMAT = "%Y%m%d%H"

# Set up array of files
filenames = []
# # Populate the allowed dates
print("Finding dates")
while idate != "1992010106":
    filenames.append(idate)
    newdate = datetime.strptime(idate, SPEEDY_DATE_FORMAT) + delta
    idate = newdate.strftime(SPEEDY_DATE_FORMAT)

print("all dates found!")

n_files = len(filenames)
print(n_files)
output_dir = os.path.join(analysis_root, 'annual')

full_fields = False

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

#################### Hybrid ####################
if HYBRID:
    print("Start HYBRID")
    # Set the directory where the Fortran binary files are located
    if full_fields:
        ds_t, ds_q, ds_ps, ds_precip = read_files_2_nc(os.path.join(HYBRID_data_root, GP_name), n_files)
        ds_precip.to_netcdf(os.path.join(output_dir,'HYBRID_precip.nc'), engine='netcdf4')
        ds_ps.to_netcdf(os.path.join(output_dir,'HYBRID_ps.nc'), engine='netcdf4')
        ds_t.to_netcdf(os.path.join(output_dir,'HYBRID_T.nc'), engine='netcdf4')
        ds_q.to_netcdf(os.path.join(output_dir,'HYBRID_Q.nc'), engine='netcdf4')
    else:
        ds_precip = read_files_2_precip_nc(os.path.join(HYBRID_data_root, GP_name), n_files)
        # Write the Dataset to a netCDF4 file
        ds_precip.to_netcdf(os.path.join(output_dir,'HYBRID_precip.nc'), engine='netcdf4')
    # ds_ps.to_netcdf(os.path.join(output_dir,'HYBRID_ps.nc'), engine='netcdf4')
    # ds_t.to_netcdf(os.path.join(output_dir,'HYBRID_T.nc'), engine='netcdf4')
    # ds_q.to_netcdf(os.path.join(output_dir,'HYBRID_Q.nc'), engine='netcdf4')
#################### SPEEDY ####################
if SPEEDY:
    print("Start SPEEDY")
    # Set the directory where the Fortran binary files are located
    if full_fields:
        ds_t, ds_q, ds_ps, ds_precip = read_files_2_nc(os.path.join(SPEEDY_root), n_files)
        ds_precip.to_netcdf(os.path.join(output_dir,'SPEEDY_precip.nc'), engine='netcdf4')
        ds_ps.to_netcdf(os.path.join(output_dir,'SPEEDY_ps.nc'), engine='netcdf4')
        ds_t.to_netcdf(os.path.join(output_dir,'SPEEDY_T.nc'), engine='netcdf4')
        ds_q.to_netcdf(os.path.join(output_dir,'SPEEDY_Q.nc'), engine='netcdf4')
    else:
        ds_precip = read_files_2_precip_nc(os.path.join(SPEEDY_root), n_files)
        # Write the Dataset to a netCDF4 file
        ds_precip.to_netcdf(os.path.join(output_dir,'SPEEDY_precip.nc'), engine='netcdf4')
    # ds_ps.to_netcdf(os.path.join(output_dir,'SPEEDY_ps.nc'), engine='netcdf4')
    # ds_t.to_netcdf(os.path.join(output_dir,'SPEEDY_T.nc'), engine='netcdf4')
    # ds_q.to_netcdf(os.path.join(output_dir,'SPEEDY_Q.nc'), engine='netcdf4')

