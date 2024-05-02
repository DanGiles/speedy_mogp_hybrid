#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List
import xarray as xr
import sys


def read_grd(filename) -> np.ndarray:
    nlon = 96
    nlat = 48
    nlev = 8
    nrec = 10
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data

def read_flx(filename) -> np.ndarray:
    nlon = 96
    nlat = 48
    f = np.fromfile(filename, dtype=np.float64)
    # shape = (nlon, nlat, nrec)
    shape = (nlon, nlat, -1)
    data = np.reshape(f, shape, order="F")
    # data = data.astype(np.float64)
    return data


def read_flux_files_2_nc(folder):
    # Loop through all the Fortran binary files
    ds_cc = xr.Dataset()
    ds_tsr = xr.Dataset()
    ds_olr = xr.Dataset()

    cloudc = []
    tsr = []
    olr = []
    i = 0

    # for date in filenames:              
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file and if correct date
        if os.path.isfile(f) and "grd" in f and "_fluxes.grd" in f:
            # Create a DataArray for each level along the 'z' dimension
            fdata = read_flx(f)
            cloudc.append(fdata[:, :, 0])
            tsr.append(fdata[:, :, 4])
            olr.append(fdata[:, :, 5])
            i += 1
    print("Number of files read = ", i)

    # Set up coordinates
    lon = np.linspace(-180, 180, 96, endpoint=True)
    lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
    lat = np.array([float(val) for val in lat_vals.split()])
    coords = {'timestamp': range(i), 'longitude': lon, 'latitude': lat}
    dims = ['timestamp', 'longitude', 'latitude']


    var_name = f'cloudc'
    ds_cc = xr.DataArray(np.stack(cloudc, axis=0), coords=coords, dims=dims, name=var_name)
    ds_cc[var_name] = ds_cc

    var_name = f'tsr'
    ds_tsr = xr.DataArray(np.stack(tsr, axis=0), coords=coords, dims=dims, name=var_name)
    ds_tsr[var_name] = ds_tsr

    var_name = f'olr'
    ds_olr = xr.DataArray(np.stack(olr, axis=0), coords=coords, dims=dims, name=var_name)
    ds_olr[var_name] = ds_olr
    
    return ds_cc, ds_tsr, ds_olr


def read_files_2_nc(folder):
    # Loop through all the Fortran binary files
    ds_t = xr.Dataset()
    ds_q = xr.Dataset()
    ds_ps = xr.Dataset()
    ds_precip = xr.Dataset()

    ps = []
    precip = []
    T = [[] for _ in range(8)]
    Q = [[] for _ in range(8)]
    i = 0
    # for date in filenames:              
    for filename in sorted(os.listdir(folder)):
        f = os.path.join(folder, filename)
        # checking if it is a file and if correct date
        if os.path.isfile(f) and "grd" in f and "fluxes.grd" not in f:
            # Create a DataArray for each level along the 'z' dimension
            fdata = read_grd(f)
            ps.append(fdata[:, :, 32])
            precip.append(fdata[:, :, 33])
            for z in range(8):
                T[z].append(fdata[:, :, z+16])
            for z in range(8):
                Q[z].append(fdata[:, :, z+24])
            i += 1
    print("Number of files read = ", i)
    # Set up coordinates
    lon = np.linspace(-180, 180, 96, endpoint=True)
    lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
    lat = np.array([float(val) for val in lat_vals.split()])

    coords = {'timestamp': range(i), 'longitude': lon, 'latitude': lat}
    dims = ['timestamp', 'longitude', 'latitude']

    # Create a DataArray for each level along 'z' and assign it to the Dataset
    for z in range(8):
        var_name = f'T_{z}'
        da_t = xr.DataArray(np.stack(T[z], axis=0), coords=coords, dims=dims, name=var_name)
        ds_t[var_name] = da_t

    for z in range(8):
        var_name = f'Q_{z}'
        da_q = xr.DataArray(np.stack(Q[z], axis=0), coords=coords, dims=dims, name=var_name)
        ds_q[var_name] = da_q

    var_name = f'ps'
    da_ps = xr.DataArray(np.stack(ps, axis=0), coords=coords, dims=dims, name=var_name)
    ds_ps[var_name] = da_ps

    var_name = f'precip'
    da_precip = xr.DataArray(np.stack(precip, axis=0), coords=coords, dims=dims, name=var_name)
    ds_precip[var_name] = da_precip
    
    return ds_t, ds_q, ds_ps, ds_precip


def main(data_root):
    GP_name = 'gp_with_oro_var'

    if "nature" not in data_root:
        print("Reading in Hybrid")
        name = "HYBRID"
        file_folder = os.path.join(data_root, GP_name)
        output_dir = os.path.join(data_root, 'annual')
    else:
        print("Reading in SPEEDY")
        name = "SPEEDY"
        file_folder = data_root
        output_dir = os.path.join(data_root, 'annual')
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ds_cc, ds_tsr, ds_olr = read_flux_files_2_nc(file_folder)
    # # Write the Dataset to a netCDF4 file
    ds_cc.to_netcdf(os.path.join(output_dir,f'{name}_cloudc.nc'), engine='netcdf4')
    ds_tsr.to_netcdf(os.path.join(output_dir,f'{name}_tsr.nc'), engine='netcdf4')
    ds_olr.to_netcdf(os.path.join(output_dir,f'{name}_olr.nc'), engine='netcdf4')

    ds_t, ds_q, ds_ps, ds_precip = read_files_2_nc(file_folder)
    ds_precip.to_netcdf(os.path.join(output_dir,f'{name}_precip.nc'), engine='netcdf4')
    ds_ps.to_netcdf(os.path.join(output_dir,f'{name}_ps.nc'), engine='netcdf4')
    ds_t.to_netcdf(os.path.join(output_dir,f'{name}_T.nc'), engine='netcdf4')
    ds_q.to_netcdf(os.path.join(output_dir,f'{name}_Q.nc'), engine='netcdf4')


if __name__ == '__main__':
    main(sys.argv[1])