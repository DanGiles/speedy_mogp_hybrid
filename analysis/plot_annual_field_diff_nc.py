import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy
import xarray as xr

# from script_variables import *

hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/run_1"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy_myriad/annual"
##### COMMENT OUT ONE LINE AS NEEDED #####
# neutral_or_warm = 'warm'
neutral_or_warm = 'neutral'
#################################


output_path = os.path.join(hybrid_path)

# Comment out variables to exclude
vars = {
    'precip': ['Precipitation', 'mm/day'],
    'olr': ['Outgoing longwave radiation', 'Upward W/m^2'], 
    'tsr': ['Top-of-atm. shortwave radiation', 'Downward W/m^2'], 
    'cloudc': ['Total cloud cover', 'Fraction']
}


def plot_map_mask(ax, field_data, t_stats, vmin, vmax, title, unit, cmap, aspect) -> None:
    ax.coastlines()
    vabsmax = max(np.abs(vmin), np.abs(vmax))

    desired_ticks = [0, 90, 180, 270]

    # Transform the tick locations to the projection coordinates
    projection = ccrs.PlateCarree(central_longitude=180)
    projected_ticks = [projection.transform_point(x, 0, ccrs.Geodetic()) for x in desired_ticks]

    # Set the x-ticks using the projected coordinates
    ax.set_xticks([pt[0] for pt in projected_ticks])
    ax.set_xticklabels(desired_ticks)

    mask = np.abs(t_stats) < 2.0
    field_mask = np.ma.array(field_data, mask=mask)

    heatmap = ax.contourf(
        lon_grid, 
        lat_grid, 
        field_mask,
        cmap=cmap, 
        vmin=-vabsmax,
        vmax=vabsmax,
        norm=mpl.colors.CenteredNorm(),
        transform=ccrs.PlateCarree()

    )
    ax.set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    ax.set_title(title)
    cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', aspect=aspect, pad = 0.2)
    cbar.ax.set_xlabel(f'{unit}')
    return heatmap

nlon = 96
nlat = 48
lon = np.linspace(0, 360, nlon, endpoint=False) # endpoint=False to match SPEEDY, but doesn't include UK
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)

runs = ['HYBRID', 'SPEEDY']
n_samples = (3652*4)

fig, axes = plt.subplots(
        nrows=2, 
        ncols=2, 
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )
# fig.suptitle('Annual Differences: HYBRID minus SPEEDY')
for i, field in enumerate(vars):
    title, units = vars[field]

    hybrid = xr.load_dataset(os.path.join(hybrid_path, f'HYBRID_{field}.nc'))[field]
    speedy = xr.load_dataset(os.path.join(speedy_path, f'SPEEDY_{field}.nc'))[field]

    hybrid_mean = hybrid.mean("timestamp")
    speedy_mean = speedy.mean("timestamp")
    hybrid_mean = hybrid_mean.assign_coords(longitude=lon)
    speedy_mean = speedy_mean.assign_coords(longitude=lon)

    hybrid_std = hybrid.std("timestamp")
    speedy_std = speedy.std("timestamp")
    hybrid_std = hybrid_std.assign_coords(longitude=lon)
    speedy_std = speedy_std.assign_coords(longitude=lon)

    diff = hybrid_mean - speedy_mean
    t_test_statistics = diff/np.sqrt((hybrid_std**2 + speedy_std**2)/n_samples)
    vmin = np.min(diff)
    vmax = np.max(diff)
    cmap = mpl.cm.PuOr_r
    if field in ['precip', 'precls', 'precnv', 'sprecnv', 'sprecls']:
        cmap = mpl.cm.PuOr
    scalarmap = plot_map_mask(axes.flat[i], diff.T, t_test_statistics.T, vmin, vmax, f'{title}', units, cmap, 25)
    if field == 'precip' and neutral_or_warm == 'neutral':
        x_low = (180 - 168.75) #11.25 #could change to 30/35 if desired
        x_high = (180 - 78.75) # 101.25
        y_low = 0 # stands out more than 90
        y_high = 15
        axes.flat[i].vlines(x_low, y_low, y_high, color='red')
        axes.flat[i].vlines(x_high, y_low, y_high, color='red')
        axes.flat[i].hlines(y_high, x_low, x_high, color='red')
        axes.flat[i].hlines(y_low, x_low, x_high, color='red')
        axes.flat[i].plot((33.75-180), -1.856, '*', color='r')
plt.suptitle("Hybrid - SPEEDY")
plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.savefig(os.path.join(output_path, f'hybrid_speedy_diffs_masked_{neutral_or_warm}.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()