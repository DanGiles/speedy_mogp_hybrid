import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from typing import List

from script_variables import *

def round_nearest_half(x):
    return round(x * 2.0)/2

nlon = 96
nlat = 48
nlev = 8

# Set up the coordinate system
lons = np.linspace(0, 360, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lats = np.array([float(val) for val in lat_vals.split()])

poi = [(9, 23), (8, 24), (21, 26) , (21, 25) , (22, 25), 
       (28, 24), (29, 24), (30, 24), (70, 26), (71, 26), 
       (72, 26), (73, 26), (74, 26), (75, 26), (90, 23), 
       (93, 25)]

output_dir = os.path.join(analysis_root, 'annual', 'global_precip')
for loc in poi:
    lon, lat = loc[0], loc[1]
    precip_SPEEDY = np.load(os.path.join(output_dir, f'SPEEDY_precip_{lon}_{lat}.npy'))
    precip_HYBRID = np.load(os.path.join(output_dir, f'HYBRID_precip_{lon}_{lat}.npy'))

    precip_SPEEDY = precip_SPEEDY[precip_SPEEDY > 1.0]
    precip_HYBRID = precip_HYBRID[precip_HYBRID > 1.0]

    bin_min = round_nearest_half(min(np.min(precip_SPEEDY), np.min(precip_HYBRID)))
    bin_max = round_nearest_half(max(np.max(precip_SPEEDY), np.max(precip_HYBRID)))
    bin_points = np.arange(bin_min, bin_max, 0.5)

    ############### PLOTTING ###############
    fig, ax = plt.subplots(
        1, 1, 
        figsize=(8, 8)
    )
    fig.suptitle(f'Lifted Index Histogram \n Location: {lons[lon], lats[lat]}.')

    y1, x1, _ = ax.hist(precip_SPEEDY, bins=bin_points, alpha=0.5, label="SPEEDY")
    y2, x2, _ = ax.hist(precip_HYBRID, bins=bin_points, alpha=0.5, label="Hybrid")

    y = max(max(y1), max(y2)) + 50
    ax.set_xlim(1, bin_max)
    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_ylabel("Counts")
    ax.legend()
    
    plt.savefig(
        os.path.join(output_dir, f'precip_{lon}_{lat}.png')
    )
    plt.close()
    