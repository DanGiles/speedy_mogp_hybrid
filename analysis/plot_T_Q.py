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


    if i == 0:
        cmap = cmo.cm.deep

        heatmap= ax.contourf(
            speedy_lon_grid, 
            speedy_lat_grid, 
            field_data,
            levels = boundaries,
            extend = 'max',
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )
    elif i == 1:
        cmap = cmo.cm.balance_r

        heatmap= ax.contourf(
            speedy_lon_grid, 
            speedy_lat_grid, 
            field_data,
            # levels = boundaries,
            extend = 'both',
            cmap=cmap,
            norm=mpl.colors.CenteredNorm(),
            transform=ccrs.PlateCarree()
        )
    elif i == 2 or i == 5:
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
    elif i == 3:
        cmap = cmo.cm.deep

        heatmap= ax.contourf(
            speedy_lon_grid, 
            speedy_lat_grid, 
            field_data,
            extend = 'both',
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )
    elif i == 4:
        cmap = cmo.cm.balance_r

        heatmap= ax.contourf(
            speedy_lon_grid, 
            speedy_lat_grid, 
            field_data,
            extend = 'both',
            cmap=cmap,
            norm=mpl.colors.CenteredNorm(),
            transform=ccrs.PlateCarree()
        )

    # ax.set_xticks(ticks=[0, 90, 180, 270, 360])
    ax.set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', aspect=aspect)
    cbar.ax.set_xlabel(f'{unit}')
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    ax.set_title(title)

def is_DJF(month):
    return (month >= 12) | (month <= 2)

def is_JJA(month):
    return (month >= 6) & (month <= 8)

def is_MAM(month):
    return (month >= 3) & (month <= 5)

hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/run_1"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy/annual"
ERA5_path = "/Users/dangiles/Documents/Stats/MetOffice/weather_data_processing/weatherbench"


# Set up the coordinate system
lon = np.linspace(0, 360, 96, endpoint=False)
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()])
speedy_lon_grid, speedy_lat_grid = np.meshgrid(lon, lat)

runs = ['HYBRID', 'SPEEDY']
fields = ['T', 'Q']
field_names = ['T_0', 'Q_0']

era_field_names = ['temperature', 'specific_humidity']
units = ['K', 'kg/kg']


for i, field in enumerate(fields):

    fig = plt.figure(figsize=(20, 23))
    gs = fig.add_gridspec(3, 6) 
    hybrid = xr.load_dataset(os.path.join(hybrid_path, f'{runs[0]}_{field}.nc'))[field_names[i]]
    hybrid = hybrid.rename({'timestamp': 'time'})
    hybrid = hybrid.transpose('time', 'latitude', 'longitude')
    
    speedy = xr.load_dataset(os.path.join(speedy_path, f'{runs[1]}_{field}.nc'))[field_names[i]]
    speedy = speedy.rename({'timestamp': 'time'})
    speedy = speedy.transpose('time', 'latitude', 'longitude')
    # hybrid = hybrid.mean('timestamp')
    # speedy = speedy.mean('timestamp')
    
    # print(hybrid)
    era5 = xr.open_dataset(os.path.join(ERA5_path, f"ERA_{field}.nc"))[era_field_names[i]]

    hybrid = hybrid.assign_coords(time=era5.time)
    speedy = speedy.assign_coords(time=era5.time)

    era5 = era5.sel(time=is_JJA(era5['time.month']))
    hybrid = hybrid.sel(time=is_JJA(hybrid['time.month']))
    speedy = speedy.sel(time=is_JJA(speedy['time.month']))
    
    era5 = era5.mean('time')
    speedy = speedy.mean('time')
    hybrid = hybrid.mean('time')
    era5 = era5.assign_coords(longitude=lon)
    speedy = speedy.assign_coords(longitude=lon)
    hybrid = hybrid.assign_coords(longitude=lon)
    # # 
    # speedy = speedy.T
    # hybrid = hybrid.T

    speedy_diff = speedy - era5
    hybrid_diff = hybrid - era5
    diff = abs(hybrid_diff) - abs(speedy_diff)
    # Global
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
    print("Global Weighted RMSE for speedy_diff:", weighted_speedy_rmse.values)
    print("Global Weighted RMSE for hybrid_diff:", weighted_hybrid_rmse.values)
    print("Global Weighted area percent = ", (abs(weighted_speedy_rmse.values - weighted_hybrid_rmse.values)/ weighted_speedy_rmse.values)*100)

    # Tropics
    lat_min = np.argmin(abs(lat_grid[:,0] - 66.564))
    lat_max = np.argmin(abs(lat_grid[:,0] - 113.436))
    tropic_weighted_lat = np.sin(np.deg2rad(lat_grid[lat_min+1 : lat_max, :]))
    # Compute the square of the differences
    LatIndexer = 'latitude'
    tropic_speedy_diff = speedy_diff.sel(**{LatIndexer: slice(-23.436, 23.436)})
    tropic_hybrid_diff = hybrid_diff.sel(**{LatIndexer: slice(-23.436, 23.436)})

    tropic_speedy_diff_squared = tropic_speedy_diff ** 2
    tropic_hybrid_diff_squared = tropic_hybrid_diff ** 2

    # Scale the squared differences by the sine of the latitude
    # speedy_diff_scaled = np.sum(speedy_diff_squared * weighted_lat)/np.sum(weighted_lat)
    # hybrid_diff_scaled = np.sum(hybrid_diff_squared * weighted_lat)/np.sum(weighted_lat)
    tropic_speedy_diff_scaled = (tropic_speedy_diff_squared * tropic_weighted_lat)
    tropic_hybrid_diff_scaled = (tropic_hybrid_diff_squared * tropic_weighted_lat)

    # Calculate the square root to obtain the weighted RMSE
    tropic_weighted_speedy_rmse = np.sqrt(np.mean(tropic_speedy_diff_scaled))
    tropic_weighted_hybrid_rmse = np.sqrt(np.mean(tropic_hybrid_diff_scaled))
    # weighted_speedy_rmse = np.sqrt(speedy_diff_scaled)
    # weighted_hybrid_rmse = np.sqrt(hybrid_diff_scaled)
    print("Tropics Weighted RMSE for speedy_diff:", tropic_weighted_speedy_rmse.values)
    print("Tropics Weighted RMSE for hybrid_diff:", tropic_weighted_hybrid_rmse.values)
    print("Tropics Weighted area percent = ", (abs(tropic_weighted_speedy_rmse.values - tropic_weighted_hybrid_rmse.values)/ tropic_weighted_speedy_rmse.values)*100)

    min = -1.0*np.max(speedy_diff)
    max = np.max(speedy_diff)

    print(np.min(abs(speedy_diff.values) - abs(hybrid_diff.values)), np.max(abs(speedy_diff.values) - abs(hybrid_diff.values)))
    # Create the first subplot in the top left
    ax1 = fig.add_subplot(gs[0, 0:3], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax1, speedy, 
             f'{field_names[i]} SPEEDY', 
             units[i], np.min(era5), np.max(era5), (i*3), 25)
   

    # Create the second subplot in the top right
    ax2 = fig.add_subplot(gs[0, 3:], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax2, hybrid, 
             f'{field_names[i]} Hybrid', 
             units[i], np.min(era5), np.max(era5), (i*3), 25)
    
    # Create the first subplot in the top left
    ax1 = fig.add_subplot(gs[1, 0:3], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax1, speedy_diff, 
             f'{field_names[i]} (SPEEDY - ERA5) \n Area-Weighted RMSE = {np.around(weighted_speedy_rmse.values, decimals=2)}', 
             units[i], np.min(speedy_diff), np.max(speedy_diff), (i*3)+1, 25)
   

    # Create the second subplot in the top right
    ax2 = fig.add_subplot(gs[1, 3:], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax2, hybrid_diff, 
             f'{field_names[i]} (Hybrid - ERA5) \n Area-Weighted RMSE = {np.around(weighted_hybrid_rmse.values, decimals=2)}', 
             units[i], np.min(speedy_diff), np.max(speedy_diff), (i*3)+1, 25)
    
    # Create the third subplot in the bottom middle
    ax3 = fig.add_subplot(gs[2, 1:5], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax3, abs(hybrid_diff) - abs(speedy_diff), 
             '|Hybrid - ERA5| - |SPEEDY - ERA5|', 
             units[i], np.min(hybrid_diff - speedy_diff), np.max(hybrid_diff - speedy_diff), (i*3)+2, 40)


    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.savefig(os.path.join(hybrid_path, f"ERA5_{field}_JJA_diff.png"), dpi = 300, bbox_inches='tight')
    plt.show()
