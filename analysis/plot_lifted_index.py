import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from typing import List

from script_variables import *

# SPEEDY_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy' #override for local compute, otherwise comment out
# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

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

# Set up the coordinate system
lon = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)

def get_index_mesh(lon_index_points: List[int], lat_index_points: List[int]):
    n1 = len(lon_index_points)
    n2 = len(lat_index_points)

    lon_index = lon_index_points*n2

    lat_index = []
    [lat_index.extend([index]*n1) for index in lat_index_points]

    return lon_index, lat_index, n1*n2

n_points = {}

lon_index_india_points = [17,18,19,20,21,22,23,24]
lat_index_india_points = [28,27,26]

lon_index_india, lat_index_india, n_points['india'] = get_index_mesh(
    lon_index_india_points,
    lat_index_india_points
)

lon_index_africa_points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
lat_index_africa_points = [24,23,22]

lon_index_africa, lat_index_africa, n_points['africa'] = get_index_mesh(
    lon_index_africa_points,
    lat_index_africa_points
)


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
    ax.set_aspect('auto')
    ax.set_title(title)
    return heatmap


def plot_scatter_wrapper(field_data, title):
    fig, ax = plt.subplots(
        2, 1, 
        figsize=(8, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    vmin=min(min(field_data['india']), min(field_data['africa']), -1)
    vmax=max(max(field_data['india']), max(field_data['africa']), 1)
    divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    heatmap = plot_scatter(
        ax[0],
        lon_index_india,
        lat_index_india,
        field_data['india'],
        "India",
        divnorm
    )
    heatmap = plot_scatter(
        ax[1],
        lon_index_africa,
        lat_index_africa,
        field_data['africa'],
        "Africa",
        divnorm,
        heatmap=heatmap
    )

    fig.colorbar(heatmap, ax=ax)
    fig.suptitle(f'Difference in No. of Lifted Index values < {title} \n (Hybrid - SPEEDY); Season: {season}')

    plt.savefig(
        os.path.join(output_path, f'lifted_index_scatter_{season}_{title}.png')
    )
    plt.close()


def read_const_grd(filename, nlon, nlat, var):
    # Orography = 0
    # Land/Sea Mask = 1
    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]


def plot_pcolormesh_scatter(
    ax, 
    lon_index, 
    lat_index, 
    field_data, 
    title, 
    divnorm, 
    extend=2
):
    lon_index_pcolormesh = lon_index[:]
    lat_index_pcolormesh = lat_index[:]
    for _ in range(extend):
        lon_index_pcolormesh.insert(0, min(lon_index_pcolormesh)-1)
        lat_index_pcolormesh.insert(0, max(lat_index_pcolormesh)+1)
        lon_index_pcolormesh.append(max(lon_index_pcolormesh)+1)
        lat_index_pcolormesh.append(min(lat_index_pcolormesh)-1)

    a, b, _ = get_index_mesh(
        [x + extend - lon_index[0] for x in lon_index], 
        [x + extend - lat_index[-1] for x in lat_index]
    )

    # print(lon_index, lon_index_pcolormesh)
    # print(lat_index, lat_index_pcolormesh)

    heatmap_pcolormesh = ax.pcolormesh(
        [x - lon_index_pcolormesh[0] for x in lon_index_pcolormesh],
        [x - lat_index_pcolormesh[-1] for x in lat_index_pcolormesh][::-1],
        lsm[lat_index_pcolormesh[::-1], :][:, lon_index_pcolormesh], 
        shading='auto',
        # cmap="Greys"
    )

    heatmap_scatter = ax.scatter(
        a, b, 
        c=field_data,
        s=400,
        edgecolors='k',
        cmap='PiYG',
        norm=divnorm
    )
    ax.set_aspect('auto')
    ax.set_title(title)
    return heatmap_pcolormesh, heatmap_scatter


def plot_pcolormesh_scatter_wrapper(field_data, title):
    fig, ax = plt.subplots(
        2, 1,
        figsize=(10, 8)
    )

    vmin=min(min(field_data['india']), min(field_data['africa']), -1)
    vmax=max(max(field_data['india']), max(field_data['africa']), 1)
    divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plot_pcolormesh_scatter(
        ax[0],
        lon_index_india_points,
        lat_index_india_points,
        field_data['india'],
        "India",
        divnorm
    )
    heatmap_pcolormesh, heatmap_scatter = plot_pcolormesh_scatter(
        ax[1],
        lon_index_africa_points,
        lat_index_africa_points,
        field_data['africa'],
        "Africa",
        divnorm
    )

    fig.colorbar(heatmap_pcolormesh, ax=ax, location='left', label='Sea (0.0) -> Land (1.0) Mask')
    fig.colorbar(heatmap_scatter, ax=ax)
    fig.suptitle(f'Difference in No. of Lifted Index values < {title} \n (Hybrid - SPEEDY); Season: {season}')

    plt.savefig(
        os.path.join(output_path, f'lifted_index_scatter_pcolormesh_{season}_{title}.png')
    )
    plt.close()


def plot_LI_vs_precip(counted_LI, precip, title):
    fig, ax = plt.subplots(
        1, 1,
        figsize=(6, 6)
    )

    ax.scatter(precip['india'], counted_LI['india'], label="India")
    ax.scatter(precip['africa'], counted_LI['africa'], label="Africa")

    fig.suptitle(f'Precipitation Difference vs Difference in No. of Lifted Index values < {title} \n (Hybrid - SPEEDY); Season: {season}')
    ax.legend()
    fig.supylabel(f'Difference in No. of Lifted Index values < {title}')
    fig.supxlabel('Precipitation Difference [g/(m^2 s)]')

    plt.savefig(
        os.path.join(output_path, f'lifted_index_vs_precip_{season}_{title}.png')
    )
    plt.close()


lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
lsm = np.flip(lsm.T, 0)


for season in seasons:
    print(season)

    counted_LI_m2 = {}
    counted_LI_m4 = {}
    counted_LI_m6 = {}
    # counted_LI_p2 = {}

    precip = {}

    speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"mean_precip_{season}.npy"))
    hybrid = np.load(os.path.join(analysis_root, GP_name, f"mean_precip_{season}.npy"))
    diff = hybrid - speedy
    precip['india'] = diff[lon_index_india, lat_index_india]
    precip['africa'] = diff[lon_index_africa, lat_index_africa]

    for location_i, location in enumerate(locations):
        print(location)

        speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"{location}_lifted_index_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, GP_name, f"{location}_lifted_index_{season}.npy"))

        # save the lifted index counts for <-2, <-4 and <-6.
        counted_LI_m2[location] = np.zeros((n_points[location]), dtype=int)
        counted_LI_m4[location] = np.zeros((n_points[location]), dtype=int)
        counted_LI_m6[location] = np.zeros((n_points[location]), dtype=int)
        # counted_LI_p2[location] = np.zeros((n_points[location]), dtype=int)

        for point in range(n_points[location]):
            LI_SPEEDY = speedy[point, :]
            LI_HYBRID = hybrid[point, :]

            # Count and print the number of missing points. Recall default value was set to 9999
            if np.max(LI_SPEEDY)>9000:
                # print(f'{season} - {location}, point {point+1} - SPEEDY: {np.sum(LI_SPEEDY>9000)}')
                LI_SPEEDY = LI_SPEEDY[LI_SPEEDY < 9000]
            
            if np.max(LI_HYBRID)>9000:
                # print(f'{season} - {location}, point {point+1} - Hybrid: {np.sum(LI_HYBRID>9000)}')
                LI_HYBRID = LI_HYBRID[LI_HYBRID < 9000]

            counted_LI_m2[location][point] = np.sum(LI_HYBRID <= -2) - np.sum(LI_SPEEDY <= -2)
            counted_LI_m4[location][point] = np.sum(LI_HYBRID <= -4) - np.sum(LI_SPEEDY <= -4)
            counted_LI_m6[location][point] = np.sum(LI_HYBRID <= -6) - np.sum(LI_SPEEDY <= -6)
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

            u = [np.mean(LI_SPEEDY), np.var(LI_SPEEDY), skew(LI_SPEEDY), kurtosis(LI_SPEEDY)]
            v = [np.mean(LI_HYBRID), np.var(LI_HYBRID), skew(LI_HYBRID), kurtosis(LI_HYBRID)]

            part1 = f'SPEEDY - mean: {u[0]:.3f}; var: {u[1]:.3f}; skew: {u[2]:.3f}; kurt: {u[3]:.3f}'
            part2 = f'HYBRID - mean: {v[0]:.3f}; var: {v[1]:.3f}; skew: {v[2]:.3f}; kurt: {v[3]:.3f}'

            ax.set_title(f'{part1} \n {part2}')
            ax.legend()
            
            plt.savefig(
                os.path.join(output_path, f'{location}_{point+1}_{season}_lifted_index.png')
            )
            plt.close()
        
    
    # MINUS 2
    plot_scatter_wrapper(counted_LI_m2, "-2")
    plot_pcolormesh_scatter_wrapper(counted_LI_m2, "-2")
    plot_LI_vs_precip(counted_LI_m2, precip, "-2")
    # MINUS 4
    plot_scatter_wrapper(counted_LI_m4, "-4")
    plot_pcolormesh_scatter_wrapper(counted_LI_m4, "-4")
    plot_LI_vs_precip(counted_LI_m4, precip, "-4")
    # MINUS 6
    plot_scatter_wrapper(counted_LI_m6, "-6")
    plot_pcolormesh_scatter_wrapper(counted_LI_m6, "-6")
    plot_LI_vs_precip(counted_LI_m6, precip, "-6")