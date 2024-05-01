# Easiest to run wherever mogp_emulator 0.6.1 is installed and where the trained MOGP objects are

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
# from script_variables import *

def plot_map(ax, field_data, title, unit, heatmap=None, **kwargs):
    ax.coastlines()
    heatmap = ax.contourf(lon_grid, lat_grid, field_data, **kwargs)
    # heatmap.set_clim(**kwargs)
    ax.set_xticks(ticks=[-180, -90, 0, 90, 180])
    ax.set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', aspect=25)
    cbar.ax.set_xlabel(f'{unit}')
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    ax.set_title(title)
    return heatmap

hybrid_path = "/home/dan/Documents/speedy_mogp_hybrid/results/"
oneshot_dir = os.path.join(hybrid_path, "oneshot")
# Set up the grid
nlon = 96
nlat = 48
nlev = 8
# Set the dates
# IDate = "1987060100"
dates = ["1987010100", "1987060100"]
SPEEDY_DATE_FORMAT = "%Y%m%d%H"
# Set up the coordinate system
lon = np.linspace(-180, 180, nlon, endpoint=True) 
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)

pressure_levels = [925, 850, 700, 500, 300, 200, 100, 30]

fig, axes = plt.subplots(
        nrows=2, 
        ncols=2,
        figsize=(17, 11),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )
axes = axes.flatten()

for i, date in enumerate(dates):
    variance = np.load(os.path.join(oneshot_dir, f"{date}_variance.npy"))
    # Load in the GP predictions
    T_var = variance[:8,:].T
    Q_var = variance[8:,:].T

    T_var = np.reshape(T_var, (nlon, nlat, nlev))
    Q_var = np.reshape(Q_var, (nlon, nlat, 5))
    if i == 0:
        heatmap1 = plot_map(axes[0], T_var[..., 0].T, f"{datetime.strptime(date, SPEEDY_DATE_FORMAT)}", r"$\sigma$(Temperature) [K]")
        heatmap2 = plot_map(axes[2], Q_var[..., 0].T, f"{datetime.strptime(date, SPEEDY_DATE_FORMAT)}", r"$\sigma$(Specific Humidity) [kg/kg]")
    else:
        heatmap1 = plot_map(axes[1], T_var[..., 0].T, f"{datetime.strptime(date, SPEEDY_DATE_FORMAT)}", r"$\sigma$(Temperature) [K]")
        heatmap2 = plot_map(axes[3], Q_var[..., 0].T, f"{datetime.strptime(date, SPEEDY_DATE_FORMAT)}", r"$\sigma$(Specific Humidity) [kg/kg]")

    # fig.suptitle(f'MOGP Predictions - {pressure_levels[0]}hPa \n {date}')

plt.savefig(
    os.path.join(oneshot_dir, f'gp_pred.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.close()