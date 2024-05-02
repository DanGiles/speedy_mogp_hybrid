import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy
import xarray as xr

# from script_variables import *

hybrid_path = "/home/dan/Documents/speedy_mogp_hybrid/results/run_1/annual"
speedy_path = "/home/dan/Documents/speedy_mogp_hybrid/results/speedy/annual"
##### COMMENT OUT ONE LINE AS NEEDED #####
# neutral_or_warm = 'warm'
neutral_or_warm = 'neutral'
#################################


output_path = os.path.join(hybrid_path)

# if not os.path.isdir(output_path):
#     os.mkdir(output_path)
# output_path = os.path.join(output_path, 'masked')
# if not os.path.isdir(output_path):
#     os.mkdir(output_path)
# output_path = os.path.join(output_path, neutral_or_warm)
# if not os.path.isdir(output_path):
#     os.mkdir(output_path)

# Comment out variables to exclude
vars = {
    'precip': ['Precipitation', 'mm/day'],
#     'ps': ['Air pressure', 'Pa'], 
#     'clstr': ['Stratiform cloud cover', 'fraction'], 
#     'precnv': ['Convective precipitation', 'g/(m^2 s)'], 
#     'precls': ['Large-scale precipitation', 'g/(m^2 s)'], 
    'olr': ['Outgoing longwave radiation', 'Upward W/m^2'], 
    'tsr': ['Top-of-atm. shortwave radiation', 'Downward W/m^2'], 
    'cloudc': ['Total cloud cover', 'Fraction'], 
#     'u': ['Wind speed (u)', 'm/s'], 
#     'v': ['Wind speed (v)', 'm/s'], 
#     't': ['Temperature', 'K'], 
#     'q': ['Specific humidity', 'Kg/Kg'],
#     # 'sprecnv': ['Summed convective precipitation', 'mm/day'],
#     # 'sprecls': ['Summed large-scale precipitation', 'mm/day'],
#     # 'stsr': ['Summed top-of-atm. Shortwave radiation', 'units?'],
#     # 'solr': ['Summed outgoing longwave radiation', 'units?'],
}


def plot_map_mask(ax, field_data, t_stats, vmin, vmax, title, unit, cmap, aspect) -> None:
    ax.coastlines()
    vabsmax = max(np.abs(vmin), np.abs(vmax))

    # levels = np.linspace(-vabsmax, vabsmax, 1, endpoint=True)
    # norm = mpl.colors.CenteredNorm(halfrange=vabsmax)
    # norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Using BoundaryNorm ensures colorbar is discrete
    # levels = np.linspace(-vabsmax, vabsmax, 9) #This should be an odd number
    # norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=25, extend='both')

    mask = np.abs(t_stats) < 2.0
    field_mask = np.ma.array(field_data, mask=mask)

    heatmap = ax.contourf(
        lon_grid, 
        lat_grid, 
        field_mask, 
        # levels=40, 
        # levels=levels,
        cmap=cmap, 
        # vmin=vmin, 
        # vmax=vmax, 
        vmin=-vabsmax,
        vmax=vabsmax,
        # cmap=mpl.cm.PuOr_r,
        norm=mpl.colors.CenteredNorm(),
        # extend='both', 
        # norm=norm,
    )
    # ax.xaxis.set_major_formatter(cartopy.mpl.ticker.LongitudeFormatter()) # available from cartopy v0.23
    # ax.yaxis.set_major_formatter(cartopy.mpl.ticker.LatitudeFormatter()) # available from cartopy v0.23
    ax.set_xticks(ticks=[-180, -90, 0, 90, 180])
    ax.set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    ax.set_title(title)
    cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', aspect=aspect, pad = 0.2)
    cbar.ax.set_xlabel(f'{unit}')
    return heatmap #mpl.cm.ScalarMappable(norm, cmap)


nlon = 96
nlat = 48
lon = np.linspace(-180, 180, nlon, endpoint=True) # endpoint=False to match SPEEDY, but doesn't include UK
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

    hybrid_std = hybrid.std("timestamp")
    speedy_std = speedy.std("timestamp")

    diff = hybrid_mean - speedy_mean
    t_test_statistics = diff/np.sqrt((hybrid_std**2 + speedy_std**2)/n_samples)
    vmin = np.min(diff)
    vmax = np.max(diff)
    cmap = mpl.cm.PuOr_r
    if field in ['precip', 'precls', 'precnv', 'sprecnv', 'sprecls']:
        cmap = mpl.cm.PuOr
    scalarmap = plot_map_mask(axes.flat[i], diff.T, t_test_statistics.T, vmin, vmax, f'{title}', units, cmap, 25)
    if field == 'precip' and neutral_or_warm == 'neutral':
        x_low = 25 #could change to 30/35 if desired
        x_high = 100
        y_low = -89 # stands out more than 90
        y_high = 89
        axes.flat[i].vlines(x_low, y_low, y_high, color='red')
        axes.flat[i].vlines(x_high, y_low, y_high, color='red')
        axes.flat[i].hlines(y_high, x_low, x_high, color='red')
        axes.flat[i].hlines(y_low, x_low, x_high, color='red')
    # fig.colorbar(scalarmap, ax=axes.flat[i])
plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.savefig(os.path.join(output_path, f'hybrid_speedy_diffs_masked_{neutral_or_warm}.png'), dpi=300, bbox_inches='tight')
# plt.show()
plt.close()