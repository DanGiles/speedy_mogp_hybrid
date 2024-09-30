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

desktop_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/speedy/annual"
myriad_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy_myriad/annual"
# speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/speedy/annual"
# speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy/annual"
ERA5_path = "/Users/dangiles/Documents/Stats/MetOffice/weather_data_processing/weatherbench"


# Set up the coordinate system
lon = np.linspace(0, 360, 96, endpoint=False)
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()])
speedy_lon_grid, speedy_lat_grid = np.meshgrid(lon, lat)

runs = ['SPEEDY', 'SPEEDY']
fields = ['T', 'Q']
field_names = ['T_0', 'Q_0']

era_field_names = ['temperature', 'specific_humidity']
units = ['K', 'kg/kg']


for i, field in enumerate(fields):
    fig = plt.figure(figsize=(20, 23))
    gs = fig.add_gridspec(3, 6) 
    
    desktop = xr.load_dataset(os.path.join(desktop_path, f'{runs[0]}_{field}.nc'))[field_names[i]]
    desktop = desktop.rename({'timestamp': 'time'})
    desktop = desktop.transpose('time', 'latitude', 'longitude')
    
    myriad = xr.load_dataset(os.path.join(myriad_path, f'{runs[1]}_{field}.nc'))[field_names[i]]
    myriad = myriad.rename({'timestamp': 'time'})
    myriad = myriad.transpose('time', 'latitude', 'longitude')

    era5 = xr.open_dataset(os.path.join(ERA5_path, f"ERA_{field}.nc"))[era_field_names[i]]

    desktop = desktop.isel(time=slice(0,-1))

    desktop = desktop.assign_coords(time=era5.time)
    myriad = myriad.assign_coords(time=era5.time)

    

    # era5 = era5.sel(time=is_MAM(era5['time.month']))
    # desktop = desktop.sel(time=is_MAM(desktop['time.month']))
    # myriad = myriad.sel(time=is_MAM(myriad['time.month']))
    
    era5 = era5.mean('time')
    myriad = myriad.mean('time')
    desktop = desktop.mean('time')
    era5 = era5.assign_coords(longitude=lon)
    myriad = myriad.assign_coords(longitude=lon)
    desktop = desktop.assign_coords(longitude=lon)
    # # 
    # myriad = myriad.T
    # desktop = desktop.T

    myriad_diff = myriad - era5
    desktop_diff = desktop - era5
    diff = abs(desktop_diff) - abs(myriad_diff)
    # Global
    lon_grid, lat_grid = np.meshgrid(era5.longitude, era5.latitude+90)

    weighted_lat = np.sin(np.deg2rad(lat_grid))
    # Compute the square of the differences
    myriad_diff_squared = myriad_diff ** 2
    desktop_diff_squared = desktop_diff ** 2

    # Scale the squared differences by the sine of the latitude
    # myriad_diff_scaled = np.sum(myriad_diff_squared * weighted_lat)/np.sum(weighted_lat)
    # desktop_diff_scaled = np.sum(desktop_diff_squared * weighted_lat)/np.sum(weighted_lat)
    myriad_diff_scaled = (myriad_diff_squared * weighted_lat)
    desktop_diff_scaled = (desktop_diff_squared * weighted_lat)

    # Calculate the square root to obtain the weighted RMSE
    weighted_myriad_rmse = np.sqrt(np.mean(myriad_diff_scaled))
    weighted_desktop_rmse = np.sqrt(np.mean(desktop_diff_scaled))
    # weighted_myriad_rmse = np.sqrt(myriad_diff_scaled)
    # weighted_desktop_rmse = np.sqrt(desktop_diff_scaled)
    print("Global Weighted RMSE for myriad_diff:", weighted_myriad_rmse.values)
    print("Global Weighted RMSE for desktop_diff:", weighted_desktop_rmse.values)
    print("Global Weighted area percent = ", (abs(weighted_myriad_rmse.values - weighted_desktop_rmse.values)/ weighted_myriad_rmse.values)*100)

    # Tropics
    lat_min = np.argmin(abs(lat_grid[:,0] - 66.564))
    lat_max = np.argmin(abs(lat_grid[:,0] - 113.436))
    tropic_weighted_lat = np.sin(np.deg2rad(lat_grid[lat_min+1 : lat_max, :]))
    # Compute the square of the differences
    LatIndexer = 'latitude'
    tropic_myriad_diff = myriad_diff.sel(**{LatIndexer: slice(-23.436, 23.436)})
    tropic_desktop_diff = desktop_diff.sel(**{LatIndexer: slice(-23.436, 23.436)})

    tropic_myriad_diff_squared = tropic_myriad_diff ** 2
    tropic_desktop_diff_squared = tropic_desktop_diff ** 2

    # Scale the squared differences by the sine of the latitude
    # myriad_diff_scaled = np.sum(myriad_diff_squared * weighted_lat)/np.sum(weighted_lat)
    # desktop_diff_scaled = np.sum(desktop_diff_squared * weighted_lat)/np.sum(weighted_lat)
    tropic_myriad_diff_scaled = (tropic_myriad_diff_squared * tropic_weighted_lat)
    tropic_desktop_diff_scaled = (tropic_desktop_diff_squared * tropic_weighted_lat)

    # Calculate the square root to obtain the weighted RMSE
    tropic_weighted_myriad_rmse = np.sqrt(np.mean(tropic_myriad_diff_scaled))
    tropic_weighted_desktop_rmse = np.sqrt(np.mean(tropic_desktop_diff_scaled))
    # weighted_myriad_rmse = np.sqrt(myriad_diff_scaled)
    # weighted_desktop_rmse = np.sqrt(desktop_diff_scaled)
    print("Tropics Weighted RMSE for myriad_diff:", tropic_weighted_myriad_rmse.values)
    print("Tropics Weighted RMSE for desktop_diff:", tropic_weighted_desktop_rmse.values)
    print("Tropics Weighted area percent = ", (abs(tropic_weighted_myriad_rmse.values - tropic_weighted_desktop_rmse.values)/ tropic_weighted_myriad_rmse.values)*100)

    min = -1.0*np.max(myriad_diff)
    max = np.max(myriad_diff)

    print(np.min(abs(myriad_diff.values) - abs(desktop_diff.values)), np.max(abs(myriad_diff.values) - abs(desktop_diff.values)))
    # Create the first subplot in the top left
    ax1 = fig.add_subplot(gs[0, 0:3], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax1, myriad, 
             f'{field_names[i]} myriad', 
             units[i], np.min(era5), np.max(era5), (i*3), 25)
   

    # Create the second subplot in the top right
    ax2 = fig.add_subplot(gs[0, 3:], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax2, desktop, 
             f'{field_names[i]} desktop', 
             units[i], np.min(era5), np.max(era5), (i*3), 25)
    
    # Create the first subplot in the top left
    ax1 = fig.add_subplot(gs[1, 0:3], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax1, myriad_diff, 
             f'{field_names[i]} (myriad - ERA5) \n Area-Weighted RMSE = {np.around(weighted_myriad_rmse.values, decimals=2)}', 
             units[i], np.min(myriad_diff), np.max(myriad_diff), (i*3)+1, 25)
   

    # Create the second subplot in the top right
    ax2 = fig.add_subplot(gs[1, 3:], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax2, desktop_diff, 
             f'{field_names[i]} (desktop - ERA5) \n Area-Weighted RMSE = {np.around(weighted_desktop_rmse.values, decimals=2)}', 
             units[i], np.min(myriad_diff), np.max(myriad_diff), (i*3)+1, 25)
    
    # Create the third subplot in the bottom middle
    ax3 = fig.add_subplot(gs[2, 1:5], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ax3, abs(desktop_diff) - abs(myriad_diff), 
             '|desktop - ERA5| - |myriad - ERA5|', 
             units[i], np.min(desktop_diff - myriad_diff), np.max(desktop_diff - myriad_diff), (i*3)+2, 40)


    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    # plt.savefig(os.path.join(myriad_path, f"ERA5_{field}_annaul_desktop_diff.png"), dpi = 300, bbox_inches='tight')
    # plt.show()
