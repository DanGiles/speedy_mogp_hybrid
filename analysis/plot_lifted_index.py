import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature

from script_variables import *

analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

level_925 = 0
level_850 = 1
level_700 = 2
level_500 = 3

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'lifted_index')
if not os.path.isdir(output_path):
    os.mkdir(output_path)

seasons = ['DJF', 'JJA']
locations = ['africa', 'india']

nlon = 96
nlat = 48
nlev = 8

lon = np.linspace(-180, 180, nlon)
lat = np.linspace(-90, 90, nlat)
lon_grid, lat_grid = np.meshgrid(lon, lat)

lon_index_india = [17,18,19,20,21,22,23,24,17,18,19,20,21,22,23,24,17,18,19,20,21,22,23,24]
lat_index_india = [28,28,28,28,28,28,28,28,27,27,27,27,27,27,27,27,26,26,26,26,26,26,26,26]
lon_index_africa = [i-14 for i in lon_index_india]
lat_index_africa = [i-4 for i in lat_index_india]


def round_nearest_half(x):
    return round(x * 2.0)/2


def plot_scatter(ax, lon_index, lat_index, field_data, title, divnorm, heatmap=None):
    ax.set_extent(
        [min(lon[lon_index]+180)-10, 
         max(lon[lon_index]+180)+10, 
         max(lat[lat_index])+10, 
         min(lat[lat_index])-10], 
        crs=ccrs.PlateCarree()
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    # ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    heatmap = ax.scatter(
        lon[lon_index]+180, 
        lat[lat_index], 
        c=field_data,
        s=400,
        edgecolors='k',
        cmap='PiYG',
        norm=divnorm
    )
    ax.set_title(title)
    return heatmap


for season in seasons:
    print(season)
    counted_LI_m2 = {}
    # counted_LI_p2 = {}
    for location_i, location in enumerate(locations):
        print(location)

        speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"{location}_lifted_index_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, GP_name, f"{location}_lifted_index_{season}.npy"))

        # save the lifted index counts for <-2 and >2.
        counted_LI_m2[location] = np.zeros((24), dtype=int)
        # counted_LI_p2[location] = np.zeros((24), dtype=int)

        for point in range(24):
            LI_SPEEDY = speedy[point, :]
            LI_HYBRID = hybrid[point, :]

            counted_LI_m2[location][point] = np.sum(LI_HYBRID <= -2) - np.sum(LI_SPEEDY <= -2)
            # counted_LI_p2[location][point] = np.sum(LI_HYBRID >= 2) - np.sum(LI_SPEEDY >= 2)

            bin_min = round_nearest_half(min(np.min(LI_SPEEDY), np.min(LI_HYBRID)))
            bin_max = round_nearest_half(max(np.max(LI_SPEEDY), np.max(LI_HYBRID)))
            bin_points = np.arange(bin_min, bin_max, 0.5)

            ############### PLOTTING ###############
            fig, ax = plt.subplots(
                1, 1, 
                figsize=(8, 8)
            )
            fig.suptitle(f'Lifted Index Histogram \n Season: {season} - Location: {location.capitalize()}, region {point+1}.')

            y1, x1, _ = ax.hist(LI_SPEEDY, bins=bin_points, alpha=0.5, label="SPEEDY")
            y2, x2, _ = ax.hist(LI_HYBRID, bins=bin_points, alpha=0.5, label="Hybrid")

            y = max(max(y1), max(y2)) + 50

            ax.vlines(np.mean(LI_SPEEDY), 0, y, colors='tab:blue', linestyles='dashed')
            ax.vlines(np.mean(LI_HYBRID), 0, y, colors='tab:orange', linestyles='dashed')

            ax.set_title(f'SPEEDY - mean: {np.mean(LI_SPEEDY):.3f}; var: {np.var(LI_SPEEDY):.3f} \n HYBRID - mean: {np.mean(LI_HYBRID):.3f}; var: {np.var(LI_HYBRID):.3f}')
            ax.legend()
            
            plt.savefig(
                os.path.join(output_path, f'{location}_{point+1}_{season}_lifted_index.png')
            )
            plt.close()
        

    fig, ax = plt.subplots(
        2, 1, 
        figsize=(8, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    vmin=min(min(counted_LI_m2['india']), min(counted_LI_m2['africa']))
    vmax=max(max(counted_LI_m2['india']), max(counted_LI_m2['africa']))
    divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    heatmap = plot_scatter(
        ax[0],
        lon_index_india,
        lat_index_india,
        counted_LI_m2['india'],
        "India",
        divnorm
    )
    heatmap = plot_scatter(
        ax[1],
        lon_index_africa,
        lat_index_africa,
        counted_LI_m2['africa'],
        "Africa",
        divnorm,
        heatmap=heatmap
    )

    fig.colorbar(heatmap, ax=ax)
    fig.suptitle(f'Difference in No. of Lifted Index values < -2 \n (Hybrid - SPEEDY); Season: {season}')

    plt.savefig(
        os.path.join(output_path, f'lifted_index_scatter_{season}.png')
    )
    plt.close()