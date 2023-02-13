#!/bin/bash
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

def read_flx(filename, nlon, nlat):
    f = np.fromfile(filename, dtype=np.float64)
    shape = (nlon,nlat)
    data = np.reshape(f, shape, order="F")
    # data = data.astype(np.float64)
    return data


def loop_through(folder, filenames, olr, nlon, nlat):
    i = 0
    for date in filenames:
        for file in os.scandir(folder):
            f = os.path.join(folder, file)
            # checking if it is a file and if correct date
            if os.path.isfile(f) and date in f and "_fluxes.grd" in f:
                # print(f)
                data = read_flx(f, nlon, nlat)
                # print(np.amax(data))
                olr[:,:,i] = data[:,:]
                i = i + 1
    return olr

# Directories
mogp_folder = os.path.join("/home/ucakdpg/Scratch/mogp-speedy/", "DATA")
# Dimensions of the grid
nlon = 96
nlat = 48
nlev = 8
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
olr_summer = np.zeros((nlon, nlat, len(filenames_summer)))
olr_winter = np.zeros((nlon, nlat, len(filenames_winter)))
# Loop through the files and read in the precip data
olr_summer = loop_through(mogp_folder, filenames_summer, olr_summer, nlon, nlat)
olr_winter = loop_through(mogp_folder, filenames_winter, olr_winter, nlon, nlat)
# # Save the output
np.save("/home/ucakdpg/Scratch/mogp-speedy/analysis/olr_JJA_nature.npy", olr_summer)
np.save("/home/ucakdpg/Scratch/mogp-speedy/analysis/olr_DJF_nature.npy", olr_winter)

