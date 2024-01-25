import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

output_path = os.path.join(pngs_root, 'gp_comparison')

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

thresh = 1/3
nodes = [0, thresh, 2*thresh, 1.0]
colors = ["blue", "white", "white", "red"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", list(zip(nodes, colors)))
cmap.set_under("blue")
cmap.set_over('red')

def plot_map(ax, field_data, title) -> None:
    ax.coastlines()
    heatmap= ax.contourf(
        lon_grid, 
        lat_grid, 
        field_data, 
        cmap=mpl.cm.PuOr_r,
        norm=mpl.colors.CenteredNorm()
    )
    cbar = plt.colorbar(heatmap)
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
    cbar = plt.colorbar(heatmap, cmap=cmap, ticks=[-6, -4, -2, 0, 2, 4, 6])
    ax.set_title(title)



if not os.path.isdir(output_path):
    os.mkdir(output_path)

for season in seasons:
    if season == 'DJF':
        n_samples = 3608
    elif season == 'JJA':
        n_samples = 3680

    for var, info in vars.items():
        with_mean = np.load(os.path.join(analysis_root, 'gp_with_oro_var', f"mean_{var}_{season}.npy"))
        without_mean = np.load(os.path.join(analysis_root, 'gp_without_oro_var', f"mean_{var}_{season}.npy"))

        with_var = np.load(os.path.join(analysis_root, 'gp_with_oro_var', f"var_{var}_{season}.npy"))
        without_var = np.load(os.path.join(analysis_root, 'gp_without_oro_var', f"var_{var}_{season}.npy"))

        if with_mean.ndim == 3:
            for i in range(with_mean.shape[2]):
                fig, (ax1, ax2) = plt.subplots(
                    nrows=2, 
                    ncols=1, 
                    figsize=(8, 8),
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
                )

                diff = without_mean[..., i] - with_mean[..., i]
                s_1 = with_var[..., i]
                s_2 = without_var[..., i]

                #Pooled variance t-test
                # s_p = np.sqrt(((n_samples - 1)*s_1)*((n_samples - 1)*s_2)/(2*n_samples - 1))
                # t_test_statistics = diff/(s_p*np.sqrt(2/n_samples))

                #Welch's t-test - for different variances
                t_test_statistics = diff/np.sqrt((s_1 + s_2)/n_samples)

                plot_map(ax1, diff.T, 'Difference (gp_with_oro_var - gp_without_oro_var)')
                plot_t_test(ax2, t_test_statistics.T, 'Welch\'s t-test (pixel-wise)')
                fig.suptitle(f'{info[0]} [{info[1]}] - field difference in {season} - model level {i+1}')

                plt.savefig(
                    os.path.join(output_path, f'{var}_{season}_field_diff_level_{i+1}.png')
                )
                plt.close()
        else:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, 
                ncols=1, 
                figsize=(8, 8),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            diff = without_mean - with_mean
            s_1 = with_var
            s_2 = without_var

            #Pooled variance t-test
            # s_p = np.sqrt(((n_samples - 1)*s_1)*((n_samples - 1)*s_2)/(2*n_samples - 1))
            # t_test_statistics = diff/(s_p*np.sqrt(2/n_samples))

            #Welch's t-test - for different variances
            t_test_statistics = diff/np.sqrt((s_1 + s_2)/n_samples)
            
            plot_map(ax1, diff.T, 'Difference (gp_with_oro_var - gp_without_oro_var)')
            plot_t_test(ax2, t_test_statistics.T, 'Welch\'s t-test (pixel-wise)')
            fig.suptitle(f'{info[0]} [{info[1]}] - field difference in {season}')

            plt.savefig(
                os.path.join(output_path, f'{var}_{season}_field_diff.png')
            )
            plt.close()

            # if var == 'tsr' or var == 'olr':
            #     fig, (ax1, ax2) = plt.subplots(
            #         nrows=2, 
            #         ncols=1, 
            #         figsize=(8, 8),
            #         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            #     )

            #     diff = diff*np.cos(np.radians(lat)).reshape(1, -1)
            #     s_1 = speedy_var*np.cos(np.radians(lat)).reshape(1, -1)**2
            #     s_2 = hybrid_var*np.cos(np.radians(lat)).reshape(1, -1)**2

            #     #Pooled variance t-test
            #     # s_p = np.sqrt(((n_samples - 1)*s_1)*((n_samples - 1)*s_2)/(2*n_samples - 1))
            #     # t_test_statistics = diff/(s_p*np.sqrt(2/n_samples))

            #     #Welch's t-test - for different variances
            #     t_test_statistics = diff/np.sqrt((s_1 + s_2)/n_samples)
                
            #     plot_map(ax1, diff.T, 'Difference (Hybrid - SPEEDY)')
            #     plot_t_test(ax2, t_test_statistics.T, 'Welch\'s t-test (pixel-wise)')
            #     fig.suptitle(f'{info[0]} [{info[1]}] - latitude weighted field difference in {season}')

            #     plt.savefig(
            #         os.path.join(output_path, f'{var}_{season}_weighted_field_diff.png')
            #     )
            #     plt.close()