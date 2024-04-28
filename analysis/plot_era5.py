import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as colors


def plot_map(ax, field_data, title, unit, min , max, i) -> None:
    ax.coastlines()
    if i == 0:
        vmin = int(np.floor(min))
        vmax = int(np.ceil(max))
        diff = vmax - vmin
        boundaries = np.linspace(vmin, vmax + 1, diff*2)
    else:
        vmin = min
        vmax = max
        boundaries = np.linspace(vmin, vmax, 100)

    heatmap= ax.contourf(
        speedy_lon_grid, 
        speedy_lat_grid, 
        field_data,
        levels = boundaries,
        extend = 'both',
        cmap=mpl.cm.PuOr
    )
    ax.set_xticks(ticks=[-180, -90, 0, 90, 180])
    ax.set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel(f'{unit}')
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    ax.set_title(title)


hybrid_path = "/home/dan/Documents/speedy_mogp_hybrid/results/run_2/annual"
speedy_path = "/home/dan/Documents/speedy_mogp_hybrid/results/speedy/annual"
ERA5_path = "/home/dan/Documents/speedy_mogp_hybrid/ERA5"


# Set up the coordinate system
lon = np.linspace(-180, 180, 96, endpoint=True)
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()])
speedy_lon_grid, speedy_lat_grid = np.meshgrid(lon, lat)

runs = ['HYBRID', 'SPEEDY']
fields = ['precip', 'cloudc']
era_field = ['precipitation', 'cloudc']
field_names = ['Precipitation', 'Cloud Cover']
units = ['mm/day', 'fraction']

fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 2)

for i, field in enumerate(fields):
    hybrid = xr.load_dataset(os.path.join(hybrid_path, f'{runs[0]}_{field}.nc'))[field]
    speedy = xr.load_dataset(os.path.join(speedy_path, f'{runs[1]}_{field}.nc'))[field]
    hybrid = hybrid.mean('timestamp')
    speedy = speedy.mean('timestamp')
  
    file_name = f'{era_field[i]}_10years'
    ds = xr.open_dataset(os.path.join(ERA5_path, f"{file_name}.nc"))[era_field[i]]
    era5 = ds.mean('time')

    # print(np.max(speedy), np.max(era5))
    speedy = speedy.T
    hybrid = hybrid.T

    speedy_diff = speedy.values - era5.values
    hybrid_diff = hybrid.values - era5.values

    diff = abs(hybrid_diff) - abs(speedy_diff)
    lon_grid, lat_grid = np.meshgrid(era5.longitude, era5.latitude+90)

    weighted_lat = np.sin(np.deg2rad(lat_grid))
    print(weighted_lat)

    # Compute the square of the differences
    speedy_diff_squared = speedy_diff ** 2
    hybrid_diff_squared = hybrid_diff ** 2

    # Scale the squared differences by the sine of the latitude
    speedy_diff_scaled = np.sum(speedy_diff_squared * weighted_lat)/np.sum(weighted_lat)
    hybrid_diff_scaled = np.sum(hybrid_diff_squared * weighted_lat)/np.sum(weighted_lat)

    # Calculate the square root to obtain the weighted RMSE
    # weighted_speedy_rmse = np.sqrt(np.mean(speedy_diff_scaled))
    # weighted_hybrid_rmse = np.sqrt(np.mean(hybrid_diff_scaled))
    weighted_speedy_rmse = np.sqrt(speedy_diff_scaled)
    weighted_hybrid_rmse = np.sqrt(hybrid_diff_scaled)
    print("Weighted RMSE for speedy_diff:", weighted_speedy_rmse)
    print("Weighted RMSE for hybrid_diff:", weighted_hybrid_rmse)
    print("Weighted area percent = ", (abs(weighted_speedy_rmse - weighted_hybrid_rmse)/ weighted_speedy_rmse)*100)

    rmse_speedy = np.sqrt((np.mean(speedy_diff**2)))
    rmse_hybrid = np.sqrt((np.mean(hybrid_diff**2)))
    print("SPEEDY RMSE =", rmse_speedy)
    print("HYBRID RMSE =", rmse_hybrid)

    # w_rmse_speedy = np.sqrt((np.mean(speedy_diff*np.sin(np.deg2rad(lat_grid))))**2)
    # w_rmse_hybrid = np.sqrt((np.mean(hybrid_diff*np.sin(np.deg2rad(lat_grid))))**2)
    # w_speedy = np.mean((speedy_diff*weighted_lat)**2)
    # w_rmse_speedy = np.sqrt(w_speedy)

    # w_hybrid = np.mean((hybrid_diff*weighted_lat)**2)
    # w_rmse_hybrid = np.sqrt(w_hybrid)


    # print("SPEEDY WRMSE =", w_rmse_speedy)
    # print("HYBRID WRMSE =", w_rmse_hybrid)
    # print("weighted area RMSE = ", abs(w_rmse_speedy - w_rmse_hybrid)/ w_rmse_speedy)

    #     w_sum_speedy = np.sum(np.sqrt(speedy_diff**2)*weighted_lat)/np.sum(weighted_lat)
    #     w_sum_hybrid = np.sum(np.sqrt(hybrid_diff**2)*weighted_lat)/np.sum(weighted_lat)

    # print("SPEEDY W_sumMSE =", w_sum_speedy)
    # print("HYBRID W_sumRMSE =", w_sum_hybrid)
    # print("weighted area RMSE = ", abs(w_sum_speedy - w_sum_hybrid)/ w_sum_speedy)

    min = -1.0*np.max(speedy_diff)
    max = np.max(speedy_diff)
    # Create the first subplot in the top left
    ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax1, speedy_diff, 
             f'{field_names[i]} (SPEEDY - ERA5) \n Area-Weighted RMSE = {np.format_float_scientific(weighted_speedy_rmse, precision = 2)}', 
             units[i], min, max, i)
   

    # Create the second subplot in the top right
    ax2 = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax2, hybrid_diff, 
             f'{field_names[i]} (Hybrid - ERA5) \n Area-Weighted RMSE = {np.format_float_scientific(weighted_hybrid_rmse, precision = 2)}', 
             units[i], min, max, i)


plt.subplots_adjust(wspace=0.3, hspace=0.1)
plt.savefig(os.path.join(hybrid_path, "ERA5_diff.png"), dpi = 300, bbox_inches='tight')
plt.show()
