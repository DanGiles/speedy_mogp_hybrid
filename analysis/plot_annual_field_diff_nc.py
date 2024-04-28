import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr


def plot_map(ax, field_data, title) -> None:
    
    ax.coastlines()
    heatmap= ax.contourf(
        lon_grid, 
        lat_grid, 
        field_data, 
        cmap=mpl.cm.PuOr_r,
        norm=mpl.colors.CenteredNorm()
    )
    cbar = plt.colorbar(heatmap, ax=ax)
    ax.set_title(title)

def plot_t_test(ax, field_data, title) -> None:
    ax.coastlines()
    divnorm = mpl.colors.TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
    levels = np.linspace(-6,6,13, endpoint=True)
    heatmap = ax.contourf(
        lon_grid, 
        lat_grid, 
        field_data, 
        levels=levels, 
        cmap=cmap, 
        vmin=-6, 
        vmax=6, 
        extend='both', 
        norm=divnorm
    )
    cbar = plt.colorbar(heatmap, cmap=cmap, ax = ax, ticks=[-6, -4, -2, 0, 2, 4, 6])
    ax.set_title(title)

def plot_map_mask(ax, field_data, t_stats, vmin, vmax, title, cmap) -> None:
    ax.coastlines()
    mask = np.abs(t_stats) < 2.0
    field_mask = np.ma.array(field_data, mask=mask)

    heatmap = ax.contourf(
        lon_grid, 
        lat_grid, 
        field_mask, 
        vmin=vmin, 
        vmax=vmax, 
        cmap=mpl.cm.PuOr_r,
        norm=mpl.colors.CenteredNorm()
    )
    ax.set_title(title)
    return heatmap 


hybrid_path = "/home/dan/Documents/speedy_mogp_hybrid/results/run_1/annual"
speedy_path = "/home/dan/Documents/speedy_mogp_hybrid/results/speedy/annual"
runs = ['HYBRID', 'SPEEDY']
fields = ['precip', 'olr', 'tsr', 'cloudc']
units = ['mm/day', 'upward W/m^2', 'downward W/m^2', 'fraction']
n_samples = (3652*4)
nlon = 96
lons = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lats = np.array([float(val) for val in lat_vals.split()])
lon_grid, lat_grid = np.meshgrid(lons, lats)

thresh = 1/3
nodes = [0, thresh, 2*thresh, 1.0]
colors = ["blue", "white", "white", "red"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", list(zip(nodes, colors)))
cmap.set_under("blue")
cmap.set_over('red')

x_points = np.arange(60, 75)
y_points = np.repeat(26, len(x_points))

coordinates = [lons[x_points]-180, lats[y_points]]
fig, axes = plt.subplots(
        nrows=2, 
        ncols=2, 
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )
plt.title('Hybrid - SPEEDY')
ax = axes.flatten()
for i, field in enumerate(fields):
    hybrid = xr.load_dataset(os.path.join(hybrid_path, f'{runs[0]}_{field}.nc'))[field]
    speedy = xr.load_dataset(os.path.join(speedy_path, f'{runs[1]}_{field}.nc'))[field]

    hybrid_mean = hybrid.mean('timestamp')
    hybrid_var = hybrid.std('timestamp')
    speedy_mean = speedy.mean('timestamp')
    speedy_var = speedy.std('timestamp')

    diff = hybrid_mean - speedy_mean
    t_test_statistics = diff/np.sqrt((hybrid_var + speedy_var)/n_samples)
    vmin = np.min(diff)
    vmax = np.max(diff)
    scalarmap = plot_map_mask(ax[i], diff.T, t_test_statistics.T, vmin, vmax, f'Difference {fields[i]} [{units[i]}]', cmap)
    fig.colorbar(scalarmap, ax=ax[i])

plt.savefig(os.path.join(hybrid_path, f'hybrid_speedy_diffs_masked.png'), dpi=300, bbox_inches='tight' )
plt.show()
plt.close()

    