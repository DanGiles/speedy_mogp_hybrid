#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta


def read_grd(filename, nlon, nlat, nlev):
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data


def loop_through(folder, filenames, array, nlon, nlat, nlev):
    i = 0
    for date in filenames:
        for file in os.scandir(folder):
            f = os.path.join(folder, file)
            # checking if it is a file and if correct date
            if os.path.isfile(f) and date in f and "_fluxes.grd" not in f:
                print(f)
                data = read_grd(f, nlon, nlat, nlev)
                array[:,:,i] = data[:,:,33]
                i = i + 1
    return array

# Directories
mogp_folder = "/home/ucakdpg/Scratch/mogp-speedy/DATA"
# Dimensions of the grid
nlon = 96
nlat = 48
nlev = 8
# Set up array of files
# Set up array of files
filenames_summer = []
filenames_winter = []
# Prepare the December, Janurary and February data
winter = ["12", "01", "02"]
summer = ["06", "07", "08"]
# Time increments
delta = timedelta(hours=6)
# Initial Date
idate = "1982010100"
SPEEDY_DATE_FORMAT = "%Y%m%d%H"

# # Populate the allowed dates
while idate != "1992010100":
    if idate[4:6] in summer:
        filenames_summer.append(idate)
    if idate[4:6] in winter:
        filenames_winter.append(idate)
    newdate = datetime.strptime(idate, SPEEDY_DATE_FORMAT) + delta
    idate = newdate.strftime(SPEEDY_DATE_FORMAT)

# Set up the array
pre_summer = np.zeros((nlon, nlat, len(filenames_summer)))
pre_winter = np.zeros((nlon, nlat, len(filenames_winter)))
# Loop through the files and read in the precip data
pre_summer = loop_through(mogp_folder, filenames_summer, pre_summer, nlon, nlat, nlev)
pre_winter = loop_through(mogp_folder, filenames_winter, pre_winter, nlon, nlat, nlev)
# Save the output
np.save("/home/ucakdpg/Scratch/mogp-speedy/analysis/pre_JJA_nature.npy", pre_summer)
np.save("/home/ucakdpg/Scratch/mogp-speedy/analysis/pre_DJF_nature.npy", pre_winter)

