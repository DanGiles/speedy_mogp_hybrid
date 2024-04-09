import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from typing import List

from script_variables import *

output_dir = os.path.join(analysis_root, 'annual')

runs = ["HYBRID", "SPEEDY"]
nlon = 96
nlat = 48
nlev = 8

# Set up the coordinate system
lons = np.linspace(0, 360, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lats = np.array([float(val) for val in lat_vals.split()])

sigma_levels = [0.95, 0.835, 0.685, 0.51, 0.34, 0.20, 0.095, 0.025]
pressure_levels = [925, 850, 700, 500, 300, 200, 100, 30]
# sigma_levels = np.flip(sigma_levels)

speedy_data = xr.load_dataset(os.path.join(output_dir, f'SPEEDY_T.nc'))
hybrid_data = xr.load_dataset(os.path.join(output_dir, f'HYBRID_T.nc'))

speedy_strip = np.zeros((nlev, nlat))
hybrid_strip = np.zeros((nlev, nlat))

for lev in range(nlev):
    speedy = speedy_data[f'T_{lev}']
    speedy = speedy.mean('timestamp')
    speedy = speedy[60:75, :]
    speedy = speedy.mean('lon')
    speedy_strip[lev, :] = speedy

    hybrid = hybrid_data[f'T_{lev}']
    hybrid = hybrid.mean('timestamp')
    hybrid = hybrid[60:75, :]
    hybrid = hybrid.mean('lon')
    hybrid_strip[lev, :] = hybrid


output_path = os.path.join(output_dir, 'zonal')
fig, axes = plt.subplots(nrows=3, ncols=1)
heatmap = axes[0].contourf(
        lats,
        pressure_levels, 
        speedy_strip
    )
cbar = plt.colorbar(heatmap, ax=axes[0])
axes[0].set_title('Speedy [K]')
axes[0].set_ylabel('Pressure levels [hPa]')
axes[0].set_ylim(bottom=925., top=30.)

heatmap = axes[1].contourf(
        lats,
        pressure_levels, 
        hybrid_strip
    )
cbar = plt.colorbar(heatmap, ax=axes[1])
axes[1].set_title('Hybrid [K]')
axes[1].set_ylabel('Pressure levels [hPa]')
axes[1].set_ylim(bottom=925., top=30.)

heatmap = axes[2].contourf(
        lats,
        pressure_levels, 
        (hybrid_strip - speedy_strip) 
    )
cbar = plt.colorbar(heatmap, ax=axes[2])
axes[2].set_ylabel('Pressure levels [hPa]')
axes[2].set_xlabel('Latitude')
axes[2].set_title('Hybrid - Speedy [K]')
axes[2].set_ylim(bottom=925., top=30.)
plt.savefig(os.path.join(output_path, f'Zonal_temp_diff_pacific.png'))
plt.show()
    
