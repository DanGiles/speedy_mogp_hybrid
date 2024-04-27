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


def read_flux_files_2_nc(folder, n_files):
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


def read_files_2_nc(folder, n_files):
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


def main(HYBRID_data_root):
    n_files = (3652*4)
    output_dir = os.path.join(HYBRID_data_root, 'annual')
    GP_name = 'gp_with_oro_var'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ds_cc, ds_tsr, ds_olr = read_flux_files_2_nc(os.path.join(HYBRID_data_root, GP_name), n_files)
    # Write the Dataset to a netCDF4 file
    ds_cc.to_netcdf(os.path.join(output_dir,'HYBRID_cloudc.nc'), engine='netcdf4')
    ds_tsr.to_netcdf(os.path.join(output_dir,'HYBRID_tsr.nc'), engine='netcdf4')
    ds_olr.to_netcdf(os.path.join(output_dir,'HYBRID_olr.nc'), engine='netcdf4')

    ds_t, ds_q, ds_ps, ds_precip = read_files_2_nc(os.path.join(HYBRID_data_root, GP_name), n_files)
    ds_precip.to_netcdf(os.path.join(output_dir,'HYBRID_precip.nc'), engine='netcdf4')
    ds_ps.to_netcdf(os.path.join(output_dir,'HYBRID_ps.nc'), engine='netcdf4')
    ds_t.to_netcdf(os.path.join(output_dir,'HYBRID_T.nc'), engine='netcdf4')
    ds_q.to_netcdf(os.path.join(output_dir,'HYBRID_Q.nc'), engine='netcdf4')


if __name__ == '__main__':
    main(sys.argv[1])