#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict

from script_variables import *

def make_dir(path: str) -> None:
    #do not empty directory if it doesn't exist!
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)
    # make directory
    os.mkdir(path)

def read_grd(filename) -> np.ndarray:
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data

def loop_through_grd(folder, filenames, n) -> Dict[str, np.ndarray]:
    u = np.zeros((nlon, nlat, nlev, n)) # wind u
    v = np.zeros_like(u)                # wind v
    t = np.zeros_like(u)                # temperature
    q = np.zeros_like(u)                # specific humidity
    precip = np.zeros((nlon, nlat, n))  # precipitation
    ps = np.zeros_like(precip)           # surface pressure

    i = 0
    for date in filenames:
        for file in os.scandir(folder):
            f = os.path.join(folder, file)
            # checking if it is a file and if correct date
            if os.path.isfile(f) and date in f and "_fluxes.grd" not in f:
                data = read_grd(f)

                u[..., i] = data[..., 0:8]
                v[..., i] = data[..., 8:16]
                t[..., i] = data[..., 16:24]
                q[..., i] = data[..., 24:32]
                ps[..., i] = data[..., 32]
                precip[..., i] = data[..., 33]

                i += 1

    return {
            'u': u, 
            'v': v, 
            't': t,
            'q': q, 
            'precip': precip,
            'ps': ps,
        }


def read_flx(filename) -> np.ndarray:
    f = np.fromfile(filename, dtype=np.float64)
    # shape = (nlon, nlat, nrec)
    shape = (nlon, nlat, -1)
    data = np.reshape(f, shape, order="F")
    # data = data.astype(np.float64)
    return data

def loop_through_flx(folder, filenames, n) -> Dict[str, np.ndarray]:
    cloudc  = np.zeros((nlon, nlat, n))
    clstr   = np.zeros_like(cloudc)
    precnv  = np.zeros_like(cloudc)
    precls  = np.zeros_like(cloudc)
    tsr     = np.zeros_like(cloudc)
    olr     = np.zeros_like(cloudc)

    # sprecnv = np.zeros_like(cloudc)
    # sprecls = np.zeros_like(cloudc)
    # stsr    = np.zeros_like(cloudc)
    # solr    = np.zeros_like(cloudc)

    i = 0
    for date in filenames:
        for file in os.scandir(folder):
            f = os.path.join(folder, file)
            # checking if it is a file and if correct date
            if os.path.isfile(f) and date in f and "_fluxes.grd" in f:
                data = read_flx(f)
                
                cloudc[..., i]  = data[..., 0]
                clstr[..., i]   = data[..., 1]
                precnv[..., i]  = data[..., 2]
                precls[..., i]  = data[..., 3]
                tsr[..., i]     = data[..., 4]
                olr[..., i]     = data[..., 5]

                # sprecnv[..., i] = data[..., 6]
                # sprecls[..., i] = data[..., 7]
                # stsr[..., i]    = data[..., 8]
                # solr[..., i]    = data[..., 9]

                i += 1

    return {
            'cloudc': cloudc,
            'clstr': clstr, 
            'precnv': precnv,
            'precls': precls,
            'tsr': tsr,
            'olr': olr,
            # 'sprecnv': sprecnv,
            # 'sprecls': sprecls,
            # 'stsr': stsr,
            # 'solr': solr,
        }

# only save the field means and variances
def save_summaries(array, name, filename) -> None:
    # calculate mean
    output_mean = np.mean(array, axis=-1)
    np.save(
        os.path.join(analysis_root, name, f"mean_{filename}.npy"),
        output_mean
    )

    # calculate variance
    output_var = np.var(array, axis=-1)
    np.save(
        os.path.join(analysis_root, name, f"var_{filename}.npy"),
        output_var
    )

    # save Indian continent & indian sea points
    lon_index_india = [17,18,19,20,21,22,23,24,17,18,19,20,21,22,23,24,17,18,19,20,21,22,23,24]
    lat_index_india = [28,28,28,28,28,28,28,28,27,27,27,27,27,27,27,27,26,26,26,26,26,26,26,26]
    output_india = array[lon_index_india, lat_index_india, ...]
    np.save(
        os.path.join(analysis_root, name, f"india_{filename}.npy"),
        output_india
    )

    # save African continent points
    lon_index_africa = [i-14 for i in lon_index_india]
    lat_index_africa = [i-4 for i in lat_index_india]
    output_africa = array[lon_index_africa, lat_index_africa, ...]
    np.save(
        os.path.join(analysis_root, name, f"africa_{filename}.npy"),
        output_africa
    )





#######################################################
# Find the file names

# Dimensions of the grid
nlon = 96
nlat = 48
nlev = 8
nrec = 10

# Prepare the December, Janurary and February data
winter = ["12", "01", "02"]
summer = ["06", "07", "08"]

# Time increments
delta = timedelta(hours=6)

# Initial Date
idate = "1982010100"
SPEEDY_DATE_FORMAT = "%Y%m%d%H"

# Set up array of files
filenames_winter = []
filenames_summer = []

# # Populate the allowed dates
print("Finding dates")
while idate != "1992010100":
    if idate[4:6] in winter:
        filenames_winter.append(idate)
    elif idate[4:6] in summer:
        filenames_summer.append(idate)
    newdate = datetime.strptime(idate, SPEEDY_DATE_FORMAT) + delta
    idate = newdate.strftime(SPEEDY_DATE_FORMAT)

print("all dates found!")

n_winter = len(filenames_winter)
n_summer = len(filenames_summer)




###########################################################
# Create the npy files


# if analysis directory does not exist, make the directory
if not os.path.isdir(analysis_root):
    os.mkdir(analysis_root)

#################### SPEEDY ####################
if SPEEDY:
    print("Start SPEEDY")
    make_dir(os.path.join(analysis_root, 'SPEEDY'))
    ################################
    ######## WINTER

    print("Start winter")
    # non-fluxes
    output = loop_through_grd(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames_winter, 
        n_winter
    )
    for varname, array in output.items():
        save_summaries(array, 'SPEEDY', f"{varname}_DJF")

    #fluxes
    output = loop_through_flx(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames_winter, 
        n_winter
    )
    for varname, array in output.items():
        save_summaries(array, 'SPEEDY', f"{varname}_DJF")
    
    ################################
    ######## SUMMER

    print("Start summer")
    # non-fluxes
    output = loop_through_grd(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames_summer, 
        n_summer
    )
    for varname, array in output.items():
        save_summaries(array, 'SPEEDY', f"{varname}_JJA")

    # fluxes
    output = loop_through_flx(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames_summer, 
        n_summer
    )
    for varname, array in output.items():
        save_summaries(array, 'SPEEDY', f"{varname}_JJA")




#################### Hybrid ####################
if HYBRID:
    print("Start Hybrid")
    make_dir(os.path.join(analysis_root, GP_name))
    data_folder = os.path.join(HYBRID_data_root, GP_name)
    ################################
    ######## WINTER

    print("Start winter")
    # non-fluxes
    output = loop_through_grd(
        data_folder, 
        filenames_winter, 
        n_winter
    )
    for varname, array in output.items():
        save_summaries(array, GP_name, f"{varname}_DJF")

    # fluxes
    output = loop_through_flx(
        data_folder, 
        filenames_winter, 
        n_winter
    )
    for varname, array in output.items():
        save_summaries(array, GP_name, f"{varname}_DJF")
    
    ################################
    ######## SUMMER

    print("Start summer")
    # non-fluxes
    output = loop_through_grd(
        data_folder, 
        filenames_summer, 
        n_summer
    )
    for varname, array in output.items():
        save_summaries(array, GP_name, f"{varname}_JJA")

    # fluxes
    output = loop_through_flx(
        data_folder, 
        filenames_summer, 
        n_summer
    )
    for varname, array in output.items():
        save_summaries(array, GP_name, f"{varname}_JJA")