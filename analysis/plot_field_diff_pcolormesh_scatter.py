import os
import numpy as np
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
import matplotlib as mpl
# import cartopy.feature as cfeature
from typing import List

from script_variables import *

# SPEEDY_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy' #override for local compute, otherwise comment out
# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

vars = {
    'precip': ['Precipitation', 'g/(m^2 s)'],
    # 'ps': ['Air pressure', 'Pa'], 
    # 'cloudc': ['Total cloud cover', 'fraction'], 
    # 'clstr': ['Stratiform cloud cover', 'fraction'], 
    'precnv': ['Convective precipitation', 'g/(m^2 s)'], 
    'precls': ['Large-scale precipitation', 'g/(m^2 s)'], 
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

seasons = ['DJF', 'JJA']
locations = ['africa', 'india']

nlon = 96
nlat = 48
nlev = 8

lon = np.linspace(-180, 180, nlon)
lat = np.linspace(-90, 90, nlat)
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
        os.path.join(output_path, f'{var}_scatter_pcolormesh_{season}.png')
    )
    plt.close()




lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
lsm = np.flip(lsm.T, 0)


for season in seasons:
    print(season)
    for var, info in vars.items():
        print(var)
        precip = {}

        speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"mean_{var}_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, GP_name, f"mean_{var}_{season}.npy"))
        diff = hybrid - speedy

        precip['india'] = diff[lon_index_india, lat_index_india]
        precip['africa'] = diff[lon_index_africa, lat_index_africa]

        plot_pcolormesh_scatter_wrapper(precip, var, info[0], info[1])