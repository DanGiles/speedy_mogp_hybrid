# Easiest to run wherever mogp_emulator 0.6.1 is installed and where the trained MOGP objects are

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from script_variables import *

def plot_map(ax, field_data, title, heatmap=None, **kwargs):
    ax.coastlines()
    heatmap = ax.contourf(lon_grid, lat_grid, field_data, **kwargs)
    # heatmap.set_clim(**kwargs)
    ax.set_title(title)
    return heatmap

oneshot_dir = os.path.join(HYBRID_data_root, "oneshot")
# Set up the grid
nlon = 96
nlat = 48
nlev = 8
# Set the dates
# IDate = "1987060100"
IDate = "1987010100"

variance = np.load(os.path.join(oneshot_dir, f"{IDate}_variance.npy"))
# Load in the GP predictions
T_var = variance[:8,:].T
Q_var = variance[8:,:].T

T_var = np.reshape(T_var, (nlon, nlat, nlev))
Q_var = np.reshape(Q_var, (nlon, nlat, 5))

# Set up the coordinate system
lon = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)

pressure_levels = [925, 850, 700, 500, 300, 200, 100, 30]

for l in range(5):
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
    fig.suptitle(f'MOGP Predictions - {pressure_levels[l]}hPa \n {IDate}')
    plt.savefig(
        os.path.join(oneshot_dir, f'{IDate}_gp_pred_{pressure_levels[l]}hPa.png'),
        dpi=200
    )
    plt.close()