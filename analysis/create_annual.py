#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List

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
while idate != "1992010112":
    filenames.append(idate)
    newdate = datetime.strptime(idate, SPEEDY_DATE_FORMAT) + delta
    idate = newdate.strftime(SPEEDY_DATE_FORMAT)

print("all dates found!")

n_files = len(filenames)

output_dir = os.path.join(analysis_root, 'annual')


if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

#################### Hybrid ####################
if HYBRID:
    print("Start HYBRID")
    fields, fluxes, T, Q = loop_through_outputs(
        os.path.join(HYBRID_data_root, GP_name), 
        filenames, 
        n_files
    )
    np.save(os.path.join(output_dir, "HYBRID_annual_means.npy"), fields)
    np.save(os.path.join(output_dir, "HYBRID_annual_fluxes.npy"), fluxes)
    np.save(os.path.join(output_dir, "HYBRID_annual_T.npy"), T)
    np.save(os.path.join(output_dir, "HYBRID_annual_Q.npy"), Q)
#################### SPEEDY ####################
if SPEEDY:
    print("Start SPEEDY")
    fields, fluxes, T, Q = loop_through_outputs(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames, 
        n_files
    )
    np.save(os.path.join(output_dir, "SPEEDY_annual_means.npy"), fields)
    np.save(os.path.join(output_dir, "SPEEDY_annual_fluxes.npy"), fluxes)
    np.save(os.path.join(output_dir, "SPEEDY_annual_T.npy"), T)
    np.save(os.path.join(output_dir, "SPEEDY_annual_Q.npy"), Q)
