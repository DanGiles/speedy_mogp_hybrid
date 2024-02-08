import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

output_path = os.path.join(pngs_root, GP_name)

# Comment out variables to exclude
vars = {
    'precip': ['Precipitation', 'g/(m^2 s)'],
    'ps': ['Air pressure', 'Pa'], 
    'cloudc': ['Total cloud cover', 'fraction'], 
    'clstr': ['Stratiform cloud cover', 'fraction'], 
    'precnv': ['Convective precipitation', 'g/(m^2 s)'], 
    'precls': ['Large-scale precipitation', 'g/(m^2 s)'], 
    'tsr': ['Top-of-atm. Shortwave radiation', 'downward W/m^2'], 
    'olr': ['Outgoing longwave radiation', 'upward W/m^2'], 
    'u': ['Wind speed (u)', 'm/s'], 
    'v': ['Wind speed (v)', 'm/s'], 
    't': ['Temperature', 'K'], 
    'q': ['Specific humidity', 'Kg/Kg'],
    # 'sprecnv': ['Summed convective precipitation', 'mm/day'],
    # 'sprecls': ['Summed large-scale precipitation', 'mm/day'],
    # 'stsr': ['Summed top-of-atm. Shortwave radiation', 'units?'],
    # 'solr': ['Summed outgoing longwave radiation', 'units?'],
}
seasons = ['DJF', 'JJA']

nlon = 96
nlat = 48
nlev = 8

# Set up the coordinate system
lon = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)

def plot_map(ax, field_data, title, heatmap=None, **kwargs):
    ax.coastlines()
    heatmap = ax.contourf(lon_grid, lat_grid, field_data, **kwargs)
    # heatmap.set_clim(**kwargs)
    ax.set_title(title)
    return heatmap



if not os.path.isdir(output_path):
    os.mkdir(output_path)

for season in seasons:
    for var, info in vars.items():
        speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f"mean_{var}_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, GP_name, f"mean_{var}_{season}.npy"))

        if speedy.ndim == 3:
            for i in range(speedy.shape[2]):
                fig, axes = plt.subplots(
                    nrows=2, 
                    ncols=1, 
                    figsize=(8, 8),
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
                )

                kwargs_contourf = {}
                kwargs_contourf['vmin'] = min(np.min(speedy[..., i]), np.min(hybrid[..., i]))
                kwargs_contourf['vmax'] = max(np.max(speedy[..., i]), np.max(hybrid[..., i]))

                heatmap = plot_map(axes[0], speedy[..., i].T, 'SPEEDY', **kwargs_contourf)
                heatmap = plot_map(axes[1], hybrid[..., i].T, 'Hybrid', heatmap, **kwargs_contourf)

                fig.colorbar(heatmap, ax=axes)
                fig.suptitle(f'{info[0]} [{info[1]}] - field mean in {season} - model level {i+1}')

                plt.savefig(
                    os.path.join(output_path, f'{var}_{season}_field_means_level_{i+1}.png')
                )
                plt.close()
        else:
            fig, axes = plt.subplots(
                nrows=2, 
                ncols=1, 
                figsize=(8, 8),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            kwargs_colorbar = {}
            kwargs_contourf = {}

            kwargs_contourf['vmin'] = min(np.min(speedy), np.min(hybrid))
            kwargs_contourf['vmax'] = max(np.max(speedy), np.max(hybrid))
            if info[1] == 'fraction':
                kwargs_colorbar['ticks'] = [0, 0.2, 0.4, 0.6, 0.8, 1]
                kwargs_contourf['levels'] = np.linspace(0, 1, 10, endpoint=True)
                kwargs_contourf['vmin'] = 0
                kwargs_contourf['vmax'] = 1
            heatmap = plot_map(axes[0], speedy.T, 'SPEEDY', **kwargs_contourf)
            heatmap = plot_map(axes[1], hybrid.T, 'Hybrid', heatmap, **kwargs_contourf)

            fig.colorbar(heatmap, ax=axes, **kwargs_colorbar)
            fig.suptitle(f'{info[0]} [{info[1]}] - field mean in {season}')

            plt.savefig(
                os.path.join(output_path, f'{var}_{season}_field_means.png')
            )
            plt.close()