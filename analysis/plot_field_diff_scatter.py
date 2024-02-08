# This script plots the difference in a field between SPEEDY and the hybrid model.
# It will create a map background version and a pcolormesh background version.

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from typing import List

from script_variables import *

# SPEEDY_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy' #override for local compute, otherwise comment out
# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out
# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/from_dan' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs/pnas' #override for local compute, otherwise comment out

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

vars = {
    'precip': ['Precipitation', 'mm/day'],
    # 'ps': ['Air pressure', 'Pa'], 
    # 'cloudc': ['Total cloud cover', 'fraction'], 
    # 'clstr': ['Stratiform cloud cover', 'fraction'], 
    'precnv': ['Convective precipitation', 'g/(m^2 s)'], #Are these units correct?
    'precls': ['Large-scale precipitation', 'g/(m^2 s)'], #Are these units correct?
    # 'tsr': ['Top-of-atm. Shortwave radiation', 'downward W/m^2'], 
    # 'olr': ['Outgoing longwave radiation', 'upward W/m^2'], 
    # 'u': ['Wind speed (u)', 'm/s'], 
    # 'v': ['Wind speed (v)', 'm/s'], 
    # 't': ['Temperature', 'K'], 
    # 'q': ['Specific humidity', 'Kg/Kg'],
    # 'sprecnv': ['Summed convective precipitation', 'mm/day'],
    # 'sprecls': ['Summed large-scale precipitation', 'mm/day'],
    # 'stsr': ['Summed top-of-atm. Shortwave radiation', 'units?'],
    # 'solr': ['Summed outgoing longwave radiation', 'units?'],
}

seasons = {'DJF': 3608, 'JJA': 3680}#, 'annual': 14608} # of the form {season: n_samples}
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
lat_index_india_points = [28,27,26,25,24,23]

lon_index_india, lat_index_india, n_points['india'] = get_index_mesh(
    lon_index_india_points,
    lat_index_india_points
)

lon_index_africa_points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
lat_index_africa_points = [25,24,23,22]

lon_index_africa, lat_index_africa, n_points['africa'] = get_index_mesh(
    lon_index_africa_points,
    lat_index_africa_points
)


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
    title: str, 
    divnorm, 
    extend: int = 2,
    s: int = 400,
    edgecolors: str = None
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

    if edgecolors is not None:
        ax.scatter(
            a, b, 
            s=s,
            edgecolors=edgecolors,
            facecolors='None',
            alpha=0.5
        )

        xticks = ax.get_xticks()
        xstep = int(xticks[1] - xticks[0])
        xtick_labels = []
        xticks = []

        yticks = ax.get_yticks()
        ystep = int(yticks[1] - yticks[0])
        ytick_labels = []
        yticks = []

        for i in range(6):
            xticks.append(i*xstep)
            val = lon[lon_index_pcolormesh[i*xstep]]+180
            if val > 180:
                val -= 360
            xtick_labels.append(val)
            yticks.append(i*ystep)
            ytick_labels.append(lat[lat_index_pcolormesh[i*ystep]])
        plt.xticks(ticks=xticks, labels=np.around(xtick_labels, 2))
        plt.yticks(ticks=yticks, labels=np.around(ytick_labels[::-1], 2))
        plt.xlabel('Longitude / [degrees]')
        plt.ylabel('Latitude / [degrees]')

    heatmap_scatter = ax.scatter(
        a, b, 
        c=field_data,
        s=s,
        edgecolors='k',
        cmap='PiYG',
        norm=divnorm,
    )

    ax.set_aspect('auto')
    ax.set_title(title)
    return heatmap_pcolormesh, heatmap_scatter


def plot_pcolormesh_scatter_wrapper(field_data, var, title, units):
    fig, ax = plt.subplots(
        2, 1,
        figsize=(10, 8)
    )

    vmin=min(min(field_data['india']), min(field_data['africa']))
    vmax=max(max(field_data['india']), max(field_data['africa']))
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
    fig.suptitle(f'Difference in {title} [{units}] \n (Hybrid - SPEEDY); Season: {season}')

    plt.savefig(
        os.path.join(output_path, f'{var}_{season}_field_diff_scatter_pcolormesh.png')
    )
    plt.close()


def plot_scatter(
    ax, 
    lon_index, 
    lat_index, 
    field_data, 
    title, 
    divnorm, 
    heatmap=None
):
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


def plot_scatter_wrapper(field_data, var, title, units):
    fig, ax = plt.subplots(
        2, 1, 
        figsize=(8, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    vmin=min(min(field_data['india']), min(field_data['africa']))
    vmax=max(max(field_data['india']), max(field_data['africa']))
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
    fig.suptitle(f'Difference in {title} [{units}] \n (Hybrid - SPEEDY); Season: {season}')

    plt.savefig(
        os.path.join(output_path, f'{var}_{season}_field_diff_scatter.png')
    )
    plt.close()


lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
lsm = np.flip(lsm.T, 0)


###### Make 2x1 plots for each season, india above, africa below ######
for season in seasons.keys():
    print(season)
    for var, info in vars.items():
        print(var)
        precip = {}

        # speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"mean_{var}_{season}.npy"))
        # hybrid = np.load(os.path.join(analysis_root, GP_name, f"mean_{var}_{season}.npy"))
        speedy = np.load(os.path.join(analysis_root, 'speedy_seasonal', f"mean_{var}_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, 'hybrid_seasonal', f"mean_{var}_{season}.npy"))
        diff = hybrid - speedy

        precip['india'] = diff[lon_index_india, lat_index_india]
        precip['africa'] = diff[lon_index_africa, lat_index_africa]

        plot_pcolormesh_scatter_wrapper(precip, var, info[0], info[1])
        plot_scatter_wrapper(precip, var, info[0], info[1])


###### Make one big plot filling in gaps between India and African regions of interest ######
###### Statistically insignificant points are masked out ######
lon_index_both_points = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
lat_index_both_points = [28,27,26,25,24,23,22]
lon_index_both, lat_index_both, n_points['both'] = get_index_mesh(
    lon_index_both_points,
    lat_index_both_points
)

for season in seasons.keys():
    for var, info in vars.items():
        print(var)

        # speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"mean_{var}_{season}.npy"))
        # hybrid = np.load(os.path.join(analysis_root, GP_name, f"mean_{var}_{season}.npy"))
        speedy = np.load(os.path.join(analysis_root, 'speedy_seasonal', f"mean_{var}_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, 'hybrid_seasonal', f"mean_{var}_{season}.npy"))
        speedy_var = np.load(os.path.join(analysis_root, 'speedy_seasonal', f"var_{var}_{season}.npy"))
        hybrid_var = np.load(os.path.join(analysis_root, 'hybrid_seasonal', f"var_{var}_{season}.npy"))
        diff = hybrid - speedy

        diff = diff[lon_index_both, lat_index_both]
        speedy_var = speedy_var[lon_index_both, lat_index_both]
        hybrid_var = hybrid_var[lon_index_both, lat_index_both]

        #Welch's t-test - for different variances
        t_stats = diff/np.sqrt((speedy_var + hybrid_var)/seasons[season])
        mask = np.abs(t_stats) < 2.0
        diff_mask = np.ma.array(diff, mask=mask)

        fig, ax = plt.subplots(
            1, 1,
            figsize=(11, 4),
            layout='constrained', 
        )

        vmin=min(diff)
        vmax=max(diff)
        divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        heatmap_pcolormesh, heatmap_scatter = plot_pcolormesh_scatter(
            ax,
            lon_index_both_points,
            lat_index_both_points,
            diff_mask,
            "Africa and India",
            divnorm,
            s=300,
            edgecolors='tab:gray'
        )

        fig.colorbar(heatmap_pcolormesh, ax=ax, location='left', label='Sea (0.0) -> Land (1.0) Mask', pad=0.005)
        fig.colorbar(heatmap_scatter, ax=ax, pad=0.005)
        fig.suptitle(f'Difference in {info[0]} [{info[1]}] \n Hybrid minus SPEEDY - Season: {season}')

        plt.savefig(
            os.path.join(output_path, f'{var}_{season}_field_diff_scatter_masked.png')
        )
        plt.close()