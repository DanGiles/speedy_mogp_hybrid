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


def get_index_mesh(lon_index_points: List[int], lat_index_points: List[int]):
    n1 = len(lon_index_points)
    n2 = len(lat_index_points)

    lon_index = lon_index_points*n2

    lat_index = []
    [lat_index.extend([index]*n1) for index in lat_index_points]

    return lon_index, lat_index


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
    lon_index_india_points = [17,18,19,20,21,22,23,24]
    lat_index_india_points = [28,27,26,25,24,23]

    lon_index_india, lat_index_india = get_index_mesh(
        lon_index_india_points,
        lat_index_india_points
    )

    output_india = array[lon_index_india, lat_index_india, ...]
    np.save(
        os.path.join(analysis_root, name, f"india_{filename}.npy"),
        output_india
    )

    # save African continent points
    lon_index_africa_points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    lat_index_africa_points = [25,24,23,22]

    lon_index_africa, lat_index_africa = get_index_mesh(
        lon_index_africa_points,
        lat_index_africa_points
    )
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
while idate != "1992010100":
    filenames.append(idate)
    newdate = datetime.strptime(idate, SPEEDY_DATE_FORMAT) + delta
    idate = newdate.strftime(SPEEDY_DATE_FORMAT)

print("all dates found!")

n_files = len(filenames)

#################### Hybrid ####################
if HYBRID:
    print("Start Hybrid")
    # non-fluxes
    output = loop_through_grd(
        os.path.join(HYBRID_data_root, GP_name),
        filenames, 
        n_files
    )
    for varname, array in output.items():
        save_summaries(array, GP_name, f"HYBRID_{varname}_annual")

    # fluxes
    output = loop_through_flx(
        os.path.join(HYBRID_data_root, GP_name), 
        filenames, 
        n_files
    )
    for varname, array in output.items():
        save_summaries(array, GP_name, f"HYBRID_{varname}_annual")

#################### SPEEDY ####################
if SPEEDY:
    print("Start SPEEDY")
    # non-fluxes
    output = loop_through_grd(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames, 
        n_files
    )
    for varname, array in output.items():
        save_summaries(array, 'annual', f"SPEEDY_{varname}_annual")

    #fluxes
    output = loop_through_flx(
        os.path.join(SPEEDY_root, "DATA", "nature"), 
        filenames, 
        n_files
    )
    for varname, array in output.items():
        save_summaries(array, 'annual', f"SPEEDY_{varname}_annual")

