import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from typing import List


hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/run_1"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy_myriad/annual"

runs = ["HYBRID", "SPEEDY"]
nlev = 8
sigma_levels = [0.95, 0.835, 0.685, 0.51, 0.34, 0.20, 0.095, 0.025]
pressure_levels = [925, 850, 700, 500, 300, 200, 100, 30]
# sigma_levels = np.flip(sigma_levels)

speedy_data = xr.load_dataset(os.path.join(speedy_path, f'SPEEDY_T.nc'))
hybrid_data = xr.load_dataset(os.path.join(hybrid_path, f'HYBRID_T.nc'))

speedy = speedy_data.mean('timestamp')
# print(speedy.longitude)
# fig = plt.figure(1)
# ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
# ax.contourf(speedy.longitude, speedy.latitude, speedy['precip'].T)
# ax.vlines(11.25, -30, 30, color='red')
# ax.vlines(101.25, -30, 30, color='red')
# ax.hlines(30, 11.25, 101.25, color='red')
# ax.hlines(-30, 11.25, 101.25, color='red')
# ax.coastlines()
# ax.set_xlabel(speedy.longitude)
# ax.set_ylabel(speedy.latitude)

# speedy = speedy.sel(latitude=slice(-30, 30))
# speedy = speedy.sel(longitude=slice(11.25, 101.25))

# fig = plt.figure(2)
# ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
# ax.contourf(speedy.longitude, speedy.latitude, speedy['precip'].T)
# ax.vlines(11.25, -30, 30, color='red')
# ax.vlines(101.25, -30, 30, color='red')
# ax.hlines(30, 11.25, 101.25, color='red')
# ax.hlines(-30, 11.25, 101.25, color='red')
# ax.coastlines()
# ax.set_xlabel(speedy.longitude)
# ax.set_ylabel(speedy.latitude)
# plt.show()

hybrid = hybrid_data.mean('timestamp')
hybrid = hybrid.sel(latitude=slice(-30, 30))
hybrid = hybrid.sel(longitude=slice(11.25, 101.25))

speedy = speedy_data.mean('timestamp')
speedy = speedy.sel(latitude=slice(-30, 30))
speedy = speedy.sel(longitude=slice(11.25, 101.25))

hybrid_strip = hybrid.mean('longitude')
speedy_strip = speedy.mean('longitude')

print(hybrid_strip)

hybrid_level = np.zeros((nlev, len(speedy.latitude)))
speedy_level = np.zeros((nlev, len(speedy.latitude)))

for lev in range(nlev):
    hybrid_level[lev, :] = hybrid_strip[f"T_{lev}"]
    speedy_level[lev, :] = speedy_strip[f"T_{lev}"]


np.save(os.path.join(speedy_path, 'Speedy_Hadley_T.npy'), speedy_level)
np.save(os.path.join(hybrid_path, 'Hybrid_Hadley_T.npy'), hybrid_level)


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
ax = axes.flatten()
heatmap = ax[0].contourf(
        speedy.latitude,
        pressure_levels, 
        (speedy_level)
    )
cbar = plt.colorbar(heatmap, ax=ax[0])
cbar.ax.tick_params(labelsize=15)
ax[0].set_ylabel('Pressure levels [hPa]', fontsize=20)
ax[0].set_xlabel('Latitude', fontsize=20)
ax[0].set_title('Speedy [K]', fontsize=20)
ax[0].tick_params(axis='both', labelsize=20)
ax[0].set_ylim(bottom=925., top=30.)

heatmap = ax[1].contourf(
        speedy.latitude,
        pressure_levels, 
        (hybrid_level)
    )
cbar = plt.colorbar(heatmap, ax=ax[1])
cbar.ax.tick_params(labelsize=15)
# ax[1].set_ylabel('Pressure levels [hPa]', fontsize=20)
ax[1].set_xlabel('Latitude', fontsize=20)
ax[1].set_title('Hybrid [K]', fontsize=20)
ax[1].tick_params(axis='both', labelsize=20)
ax[1].set_ylim(bottom=925., top=30.)

heatmap = ax[2].contourf(
        speedy.latitude,
        pressure_levels, 
        (hybrid_level - speedy_level)
    )
cbar = plt.colorbar(heatmap, ax=ax[2])
cbar.ax.tick_params(labelsize=15)
# ax[2].set_ylabel('Pressure levels [hPa]', fontsize=20)
ax[2].set_xlabel('Latitude', fontsize=20)
ax[2].set_title('Hybrid - Speedy [K]', fontsize=20)
ax[2].tick_params(axis='both', labelsize=20)
ax[2].set_ylim(bottom=925., top=30.)
plt.subplots_adjust(hspace=0.5)
plt.savefig(os.path.join(hybrid_path, f'Zonal_temp_diff_pacific.png'), bbox_inches='tight')
plt.show()
    
