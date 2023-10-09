# Easiest to run wherever mogp_emulator 0.6.1 is installed and where the trained MOGP objects are

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import mogp_emulator
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from script_variables import *

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'field_predictions')
if not os.path.isdir(output_path):
    os.mkdir(output_path)


def read_grd(filename, nlon, nlat, nlev):
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data

def read_const_grd(filename, nlon, nlat, var):
    # Orography = 0
    # Land/Sea Mask = 1
    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]

def read_oro_var() -> np.ndarray:
    oro_var_data = np.zeros((96, 48))
    oro_var_data_file = os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "std_orog_for_speedy.dat")
    with open(oro_var_data_file) as f:
        for row_i in range(96):
            oro_var_data[row_i, :] = np.fromstring(f.readline().strip(), dtype=float, sep=',')
    return oro_var_data


def data_prep(data, oro, ls, nlon, nlat) -> np.ndarray:
    T_mean = data[:,:,16:24]
    Q_mean = data[:,:,24:32]
    Q_mean = np.flip(Q_mean, axis = 2)
    T_mean = np.flip(T_mean, axis = 2)
    low_values_flags = Q_mean[:,:] < 1e-6  # Where values are low
    Q_mean[low_values_flags] = 1e-6

    if GP_name == "gp_without_oro_var":
        # Version for gp_without_oro_var
        train = np.empty(((nlon*nlat),19), dtype = np.float64)
        train[:, 0] = data[:,:,32].flatten()
        train[:, 1] = oro.flatten()
        train[:, 2] = ls.flatten()
        train[:, 3:11] = np.reshape(T_mean, ((nlon*nlat), 8))
        train[:, 11:] = np.reshape(Q_mean, ((nlon*nlat), 8))
    elif GP_name == "gp_with_oro_var":
        # Version for gp_with_oro_var
        train = np.empty(((nlon*nlat),20), dtype = np.float64)
        train[:, 0] = data[:,:,32].flatten()
        train[:, 1] = oro[...,0].flatten()
        train[:, 2] = oro[...,1].flatten() 
        train[:, 3] = ls.flatten()
        train[:, 4:12] = np.reshape(T_mean, ((nlon*nlat), 8))
        train[:, 12:] = np.reshape(Q_mean, ((nlon*nlat), 8))
    else:
        raise ValueError(f"GP_name not recognised, {GP_name} provided.")
    return train


def plot_map(ax, field_data, title, heatmap=None, **kwargs):
    ax.coastlines()
    heatmap = ax.contourf(lon_grid, lat_grid, field_data, **kwargs)
    # heatmap.set_clim(**kwargs)
    ax.set_title(title)
    return heatmap


trained_gp = pickle.load(open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "rb"))

# Defining constants and initial values
# SPEEDY_DATE_FORMAT = "%Y%m%d%H"
nature_dir = os.path.join(SPEEDY_root, "DATA", "nature")

IDate = "1982010100"
# dtDate = "1982010106"
# number_time_steps = (3652*4) 
nlon = 96
nlat = 48
nlev = 8
# dt = 6

# Initialisation steps
data = read_grd(os.path.join(nature_dir, IDate +".grd"), nlon, nlat, nlev)

# Read in the orography and land/sea fraction
oro = read_const_grd(os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "sfc.grd"), nlon, nlat, 0)
lsm = read_const_grd(os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "sfc.grd"), nlon, nlat, 1)
oro = np.flip(oro, 1)
lsm = np.flip(lsm, 1)
# rho = np.loadtxt(os.path.join(HYBRID_root, "src", "density.txt"))
if GP_name == "gp_with_oro_var":
    oro = np.stack((oro, read_oro_var()), axis=2)

test = data_prep(data, oro, lsm, nlon, nlat)
print("Data Prep")
print(test.shape)

variance, uncer, d = trained_gp.predict(test)
T_var = variance[:8,:].T
Q_var = variance[8:,:].T

T_var = np.reshape(T_var, (nlon, nlat, nlev))
Q_var = np.reshape(Q_var, (nlon, nlat, nlev))

# Set up the coordinate system
lon = np.linspace(-180, 180, nlon)
lat = np.linspace(-90, 90, nlat)
lon_grid, lat_grid = np.meshgrid(lon, lat)

pressure_levels = [30, 100, 200, 300, 500, 700, 850, 925] # hPa

for l in range(8):
    fig, axes = plt.subplots(
        nrows=2, 
        ncols=1,
        figsize=(8, 8),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )
    heatmap1 = plot_map(axes[0], T_var[..., l].T, "Temperature")
    heatmap2 = plot_map(axes[1], Q_var[..., l].T, "Specific Humidity")

    plt.colorbar(heatmap1, ax=axes[0])
    plt.colorbar(heatmap2, ax=axes[1])
    fig.suptitle(f'MOGP Predictions - {pressure_levels[l]}hPa')
    plt.savefig(
        os.path.join(output_path, f'gp_pred_{pressure_levels[l]}hPa.png'),
        dpi=200
    )
    plt.close()