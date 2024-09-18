import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import xarray as xr


############### READ ME ####################
# RUN `create_lifted_index_from_nc.py` FIRST
############################################


def round_nearest_half(x):
    return round(x * 2.0)/2

nlon = 96
nlat = 48
nlev = 8
hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/run_1/annual"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/speedy/annual"

# Set up the coordinate system
lons = np.linspace(0, 360, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lats = np.array([float(val) for val in lat_vals.split()])

# Assign a list of tuples for the points of interest. 
# Must be in the form (lon, lat). 
# More points can be added to the list.
poi = [(9, 23), (80,20)]

output_dir = os.path.join(hybrid_path, 'global_lifted_index')

precip_HYBRID = xr.load_dataset(os.path.join(hybrid_path, f'HYBRID_precip.nc'))['precip']
precip_SPEEDY = xr.load_dataset(os.path.join(speedy_path, f'SPEEDY_precip.nc'))['precip']


for loc in poi:
    lon, lat = loc[0], loc[1]
    LI_SPEEDY = np.load(os.path.join(output_dir, f'SPEEDY_LI_{lon}_{lat}.npy'))
    LI_HYBRID = np.load(os.path.join(output_dir, f'HYBRID_LI_{lon}_{lat}.npy'))
    # Count and print the number of missing points. Recall default value was set to 9999
    if np.max(LI_SPEEDY)>9000:
        # print(f'{season} - {location}, point {point+1} - SPEEDY: {np.sum(LI_SPEEDY>9000)}')
        LI_SPEEDY = LI_SPEEDY[LI_SPEEDY < 9000]
    
    if np.max(LI_HYBRID)>9000:
        # print(f'{season} - {location}, point {point+1} - Hybrid: {np.sum(LI_HYBRID>9000)}')
        LI_HYBRID = LI_HYBRID[LI_HYBRID < 9000]

    bin_min = round_nearest_half(min(np.min(LI_SPEEDY), np.min(LI_HYBRID)))
    bin_max = round_nearest_half(max(np.max(LI_SPEEDY), np.max(LI_HYBRID)))
    bin_points = np.arange(bin_min, bin_max, 0.5)

    ############### PLOTTING ###############
    fig, ax = plt.subplots(
        2, 1, 
        figsize=(17, 11)
    )
    axs = ax.flatten()
    axs[0].set_title(rf"Location ({lons[lon]}$^o$E, {lats[lat]}$^o$S)", fontsize = 30)


    u = [np.mean(LI_SPEEDY), np.var(LI_SPEEDY), skew(LI_SPEEDY), kurtosis(LI_SPEEDY)]
    v = [np.mean(LI_HYBRID), np.var(LI_HYBRID), skew(LI_HYBRID), kurtosis(LI_HYBRID)]


    y1, x1, _ = axs[0].hist(LI_SPEEDY, bins=bin_points, alpha=0.5, label=rf"SPEEDY $\mu$: {u[0]:.3f}; $\sigma^{2}$: {u[1]:.3f}")
    y2, x2, _ = axs[0].hist(LI_HYBRID, bins=bin_points, alpha=0.5, label=rf"Hybrid $\mu$: {v[0]:.3f}; $\sigma^{2}$: {v[1]:.3f}")
    y = max(max(y1), max(y2)) + 50
    axs[0].vlines(np.mean(LI_SPEEDY), 0, y, colors='tab:blue', linestyles='dashed')
    axs[0].vlines(np.mean(LI_HYBRID), 0, y, colors='tab:orange', linestyles='dashed')
    # part1 = f'SPEEDY - mean: {u[0]:.3f}; var: {u[1]:.3f}; skew: {u[2]:.3f}; kurt: {u[3]:.3f}'
    # part2 = f'HYBRID - mean: {v[0]:.3f}; var: {v[1]:.3f}; skew: {v[2]:.3f}; kurt: {v[3]:.3f}'

    # axs[0].set_title(f'{part1} \n {part2}')
    axs[0].set_xlabel("Lifted Index", fontsize=30)
    axs[0].set_ylabel("Counts", fontsize=30)
    axs[0].tick_params(axis='both', labelsize=25)
    axs[0].legend(fontsize=20)

    
    loc_speedy = precip_SPEEDY[:, lon, lat].data
    loc_speedy = loc_speedy[loc_speedy > 1.0]

    loc_hybrid = precip_HYBRID[:, lon, lat].data
    loc_hybrid = loc_hybrid[loc_hybrid> 1.0]

    bin_min = round_nearest_half(min(np.min(loc_speedy), np.min(loc_hybrid)))
    bin_max = round_nearest_half(max(np.max(loc_speedy), np.max(loc_hybrid)))
    bin_points = np.arange(bin_min, bin_max, 0.5)

    ############### PLOTTING ###############

    y1, x1, _ = axs[1].hist(loc_speedy, bins=bin_points, alpha=0.5, label="SPEEDY")
    y2, x2, _ = axs[1].hist(loc_hybrid, bins=bin_points, alpha=0.5, label="Hybrid")

    y = max(max(y1), max(y2)) + 50
    axs[1].set_xlim(1, 60)
    axs[1].tick_params(axis='both', labelsize=25)
    axs[1].set_xlabel("Precipitation (mm/day)", fontsize = 30)
    axs[1].set_ylabel("Counts", fontsize = 30)
    axs[1].legend(fontsize=20)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(
        os.path.join(output_dir, f'LI_{lon}_{lat}.png'), dpi = 300, bbox_inches='tight'
    )
    # plt.show()
    plt.close()