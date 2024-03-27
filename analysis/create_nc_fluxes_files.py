#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List
import xarray as xr

from script_variables import *

# def make_dir(path: str) -> None:
#     #do not empty directory if it doesn't exist!
#     if os.path.isdir(path):
#         import shutil
#         shutil.rmtree(path)
#     # make directory
#     os.mkdir(path)

def read_grd(filename) -> np.ndarray:
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data

def loop_through_outputs(folder, filenames, n) -> Dict[str, np.ndarray]:

    data = np.zeros((nlon, nlat, 34))
    fluxes = np.zeros((nlon, nlat, 10))
    T = np.zeros((nlon, nlat, nlev, len(filenames)+1))
    Q = np.zeros((nlon, nlat, nlev, len(filenames)+1))

    print(len(filenames))
    flux_count = 0
    count = 0

    # for date in filenames:
    #     print(date)
    for file in os.listdir(folder):
        f = os.path.join(folder, file)
        # checking if it is a file and if correct date
        if os.path.isfile(f) and ".grd" in file and "_fluxes.grd" not in file:
            print(file, count)
            fdata = read_grd(f)
            data = data + fdata
            T[..., count] = fdata[..., 16:24]
            Q[..., count] = fdata[..., 24:32]
            count = count + 1
        elif os.path.isfile(f) and "_fluxes.grd" in file:
            fdata = read_flx(f)
            fluxes = fluxes + fdata
            flux_count = flux_count + 1

    data = data/count
    fluxes = fluxes/flux_count
    return data, fluxes, T, Q

def read_flx(filename) -> np.ndarray:
    f = np.fromfile(filename, dtype=np.float64)
    # shape = (nlon, nlat, nrec)
    shape = (nlon, nlat, -1)
    data = np.reshape(f, shape, order="F")
    # data = data.astype(np.float64)
    return data


def read_files_2_nc(folder, n_files):
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

