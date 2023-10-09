import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

seasons = ['DJF', 'JJA']


# Set up the coordinate system
lon = np.linspace(-180, 180, 96)
lat = np.linspace(-90, 90, 48)
lon_grid, lat_grid = np.meshgrid(lon, lat)


def plot_quiver(ax, u, v, title, heatmap=None):
    colors = np.sqrt(u**2 + v**2)
    ax.coastlines()
    heatmap = ax.quiver(
        lon_grid, 
        lat_grid, 
        u,
        v,
        colors,
    )
    ax.set_title(title)
    return heatmap



if not os.path.isdir(output_path):
    os.mkdir(output_path)

for season in seasons:
    SPEEDY_u = np.load(os.path.join(analysis_root, GP_name, f"mean_u_{season}.npy"))
    SPEEDY_v = np.load(os.path.join(analysis_root, GP_name, f"mean_v_{season}.npy"))
    HYBRID_u = np.load(os.path.join(analysis_root, GP_name, f"mean_u_{season}.npy"))
    HYBRID_v = np.load(os.path.join(analysis_root, GP_name, f"mean_v_{season}.npy"))

    for i in range(SPEEDY_u.shape[2]):
        fig, axes = plt.subplots(
            nrows=3, 
            ncols=1, 
            figsize=(8, 8),
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
        )

        heatmap = plot_quiver(axes[0], SPEEDY_u[..., i], SPEEDY_v[..., i], 'SPEEDY')
        heatmap = plot_quiver(axes[1], HYBRID_u[..., i], HYBRID_v[..., i], 'Hyrbid', heatmap)
        heatmap2 = plot_quiver(
            axes[2], 
            HYBRID_u[..., i] - SPEEDY_u[..., i], 
            HYBRID_v[..., i] - SPEEDY_v[..., i], 
            'Difference (Hybrid - SPEEDY)'
        )

        plt.colorbar(heatmap, ax=axes[0:2], aspect=40)
        plt.colorbar(heatmap2, ax=axes[2])
        fig.suptitle(f'Wind vector [m/s] - vector field in {season} - model level {i+1}')

        plt.savefig(
            os.path.join(output_path, f'wind_vector_{season}_level_{i+1}.png'),
            dpi=200
        )
        plt.close()