import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as colors
import cmocean as cmo
from matplotlib.colors import LinearSegmentedColormap

def plot_map(ax, field_data, title, unit, min, max, i, aspect) -> None:
    ax.coastlines()
    
    vmin = int(np.floor(min))
    vmax = int(np.ceil(max))
    boundaries = np.arange(vmin, vmax + 1)


    desired_ticks = [0, 90, 180, 270]

    # Transform the tick locations to the projection coordinates
    projection = ccrs.PlateCarree(central_longitude=180)
    projected_ticks = [projection.transform_point(x, 0, ccrs.Geodetic()) for x in desired_ticks]

    # Set the x-ticks using the projected coordinates
    ax.set_xticks([pt[0] for pt in projected_ticks])
    ax.set_xticklabels(desired_ticks)

    if i == 2:
        thresh = 1/3
        nodes = [0, thresh, 2*thresh, 1.0]
        colors = ["blue", "white", "white", "red"]
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", list(zip(nodes, colors)))
        cmap.set_under("blue")
        cmap.set_over('red')
        heatmap= ax.contourf(
            speedy_lon_grid, 
            speedy_lat_grid, 
            field_data,
            # levels = boundaries,
            extend = 'both',
            cmap=mpl.cm.PuOr,
            norm=mpl.colors.CenteredNorm(),
            transform=ccrs.PlateCarree()
        )
    else:
        cmap = cmo.cm.balance_r

        heatmap= ax.contourf(
            speedy_lon_grid, 
            speedy_lat_grid, 
            field_data,
            levels = boundaries,
            extend = 'both',
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )
    # ax.set_xticks(ticks=[0, 90, 180, 270, 360])
    ax.set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', aspect=aspect)
    cbar.ax.set_xlabel(f'{unit}')
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    ax.set_title(title)


hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/run_1/annual"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy/annual"
ERA5_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/era5"


# Set up the coordinate system
lon = np.linspace(0, 360, 96, endpoint=False)
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()])
speedy_lon_grid, speedy_lat_grid = np.meshgrid(lon, lat)

runs = ['HYBRID', 'SPEEDY']
fields = ['precip']
era_field = ['precipitation']
field_names = ['Precipitation']
units = ['mm/day']

fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 6)

for i, field in enumerate(fields):
    hybrid = xr.load_dataset(os.path.join(hybrid_path, f'{runs[0]}_{field}.nc'))[field]
    speedy = xr.load_dataset(os.path.join(speedy_path, f'{runs[1]}_{field}.nc'))[field]
    hybrid = hybrid.mean('timestamp')
    speedy = speedy.mean('timestamp')
    hybrid = hybrid.assign_coords(longitude=lon)
    speedy = speedy.assign_coords(longitude=lon)
  
    file_name = f'{era_field[i]}_10years'
    ds = xr.open_dataset(os.path.join(ERA5_path, f"{file_name}.nc"))[era_field[i]]
    era5 = ds.mean('time')
    era5 = era5.assign_coords(longitude=lon)
    era5 = era5.drop_vars('step')
    era5 = era5.drop_vars('surface')
    era5 = era5.drop_vars('number')

    # print(np.max(speedy), np.max(era5))
    speedy = speedy.T
    hybrid = hybrid.T

    speedy_diff = speedy - era5
    hybrid_diff = hybrid - era5
    diff = abs(hybrid_diff) - abs(speedy_diff)
    lon_grid, lat_grid = np.meshgrid(era5.longitude, era5.latitude+90)

    weighted_lat = np.sin(np.deg2rad(lat_grid))
    # Compute the square of the differences
    speedy_diff_squared = speedy_diff ** 2
    hybrid_diff_squared = hybrid_diff ** 2

    # Scale the squared differences by the sine of the latitude
    # speedy_diff_scaled = np.sum(speedy_diff_squared * weighted_lat)/np.sum(weighted_lat)
    # hybrid_diff_scaled = np.sum(hybrid_diff_squared * weighted_lat)/np.sum(weighted_lat)
    speedy_diff_scaled = (speedy_diff_squared * weighted_lat)
    hybrid_diff_scaled = (hybrid_diff_squared * weighted_lat)

    # Calculate the square root to obtain the weighted RMSE
    weighted_speedy_rmse = np.sqrt(np.mean(speedy_diff_scaled))
    weighted_hybrid_rmse = np.sqrt(np.mean(hybrid_diff_scaled))
    # weighted_speedy_rmse = np.sqrt(speedy_diff_scaled)
    # weighted_hybrid_rmse = np.sqrt(hybrid_diff_scaled)
    print("Weighted RMSE for speedy_diff:", weighted_speedy_rmse.values)
    print("Weighted RMSE for hybrid_diff:", weighted_hybrid_rmse.values)
    print("Weighted area percent = ", (abs(weighted_speedy_rmse.values - weighted_hybrid_rmse.values)/ weighted_speedy_rmse.values)*100)

    rmse_speedy = np.sqrt((np.mean(speedy_diff**2)))
    rmse_hybrid = np.sqrt((np.mean(hybrid_diff**2)))
    print("SPEEDY RMSE =", rmse_speedy.values)
    print("HYBRID RMSE =", rmse_hybrid.values)

    min = -1.0*np.max(speedy_diff)
    max = np.max(speedy_diff)

    print(np.min(abs(speedy_diff.values) - abs(hybrid_diff.values)), np.max(abs(speedy_diff.values) - abs(hybrid_diff.values)))
    # Create the first subplot in the top left
    ax1 = fig.add_subplot(gs[0, 0:3], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax1, speedy_diff, 
             f'{field_names[i]} (SPEEDY - ERA5) \n Area-Weighted RMSE = {np.around(weighted_speedy_rmse.values, decimals=2)}', 
             units[i], -10, 10, i, 25)
   

    # Create the second subplot in the top right
    ax2 = fig.add_subplot(gs[0, 3:], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax2, hybrid_diff, 
             f'{field_names[i]} (Hybrid - ERA5) \n Area-Weighted RMSE = {np.around(weighted_hybrid_rmse.values, decimals=2)}', 
             units[i], -10, 10, i, 25)
    
    # Create the third subplot in the bottom middle
    ax3 = fig.add_subplot(gs[1, 1:5], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax3, abs(speedy_diff) - abs(hybrid_diff), 
             '|SPEEDY - ERA5| - |Hybrid - ERA5|', 
             units[i], np.min(speedy_diff - hybrid_diff), np.max(speedy_diff - hybrid_diff), 2, 40)


plt.subplots_adjust(wspace=0.3, hspace=0.1)
plt.savefig(os.path.join(hybrid_path, "ERA5_diff.png"), dpi = 300, bbox_inches='tight')
plt.show()
