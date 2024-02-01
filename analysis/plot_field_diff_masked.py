import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
# import cmocean as cmo

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/from_dan' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs/pnas' #override for local compute, otherwise comment out

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# Comment out variables to exclude
vars = {
    'precip': ['Precipitation', 'mm/day'],
    'ps': ['Air pressure', 'Pa'], 
    'cloudc': ['Total cloud cover', 'fraction'], 
    'clstr': ['Stratiform cloud cover', 'fraction'], 
    'precnv': ['Convective precipitation', 'mm/day'], 
    'precls': ['Large-scale precipitation', 'mm/day'], 
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
# seasons = ['DJF', 'JJA']

# Set up the coordinate system
nlon = 96
nlat = 48
nlev = 8

# Set up the coordinate system
lon = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)
# pressure_levels = [30, 100, 200, 300, 500, 700, 850, 925] # hPa
pressure_levels = [925, 850, 700, 500, 300, 200, 100, 30] # hPa

def plot_map_pnas_mask(ax, field_data, t_stats, vmin, vmax, title, cmap) -> None:
    ax.coastlines()
    vabsmax = max(np.abs(vmin), np.abs(vmax))

    # levels = np.linspace(vmin, vmax, 13, endpoint=True)
    # norm = mpl.colors.CenteredNorm(halfrange=vabsmax)
    # norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Using BoundaryNorm ensures colorbar is discrete
    levels = np.linspace(-vabsmax, vabsmax, 9) #This should be an odd number
    norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=256, extend='both')

    mask = np.abs(t_stats) < 2.0
    field_mask = np.ma.array(field_data, mask=mask)

    heatmap = ax.contourf(
        lon_grid, 
        lat_grid, 
        field_mask, 
        # levels=40, 
        levels=levels,
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        extend='both', 
        norm=norm,
    )
    ax.set_title(title)
    return mpl.cm.ScalarMappable(norm, cmap)

def read_files(season: str, var: str):
    if season == 'annual':
        speedy = np.load(os.path.join(analysis_root, 'annual', f"mean_SPEEDY_{var}_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, 'annual', f"mean_HYBRID_{var}_{season}.npy"))
        speedy_var = np.load(os.path.join(analysis_root, 'annual', f"var_SPEEDY_{var}_{season}.npy"))
        hybrid_var = np.load(os.path.join(analysis_root, 'annual', f"var_HYBRID_{var}_{season}.npy"))
    else:
        speedy = np.load(os.path.join(analysis_root, 'speedy_seasonal', f"mean_{var}_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, 'hybrid_seasonal', f"mean_{var}_{season}.npy"))
        speedy_var = np.load(os.path.join(analysis_root, 'speedy_seasonal', f"var_{var}_{season}.npy"))
        hybrid_var = np.load(os.path.join(analysis_root, 'hybrid_seasonal', f"var_{var}_{season}.npy"))
    return {
        'speedy': speedy, 
        'hybrid': hybrid, 
        'speedy_var': speedy_var, 
        'hybrid_var': hybrid_var,
    }


seasons = {'DJF': 3608, 'JJA': 3680, 'annual': 14608} # of the form {season: n_samples}
for var, info in vars.items():
    cmap = mpl.cm.PuOr_r
    if var in ['precip', 'precnv', 'precls', 'sprecnv', 'sprecls']:
        cmap = mpl.cm.PuOr
    (DJF, JJA, annual) = [read_files(season, var) for season in seasons.keys()]
    # (DJF, JJA, annual) = [read_files('annual', var) for _ in range(3)] # delete this line when data has been fixed
    annual = read_files('annual', var)

    if annual['speedy'].ndim == 3:
        for i in range(annual['speedy'].shape[2]):
            ####### Plot annual figure only
            fig, ax = plt.subplots(
                nrows=1, 
                ncols=1, 
                figsize=(8, 4),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            annual_diff = annual['hybrid'][..., i] - annual['speedy'][..., i]
            annual_s_1 = annual['speedy_var'][..., i]
            annual_s_2 = annual['hybrid_var'][..., i]

            #Welch's t-test - for different variances
            annual_t_test_statistics = annual_diff/np.sqrt((annual_s_1 + annual_s_2)/seasons['annual'])

            vmin = np.min(annual_diff)
            vmax = np.max(annual_diff)
            scalarmap = plot_map_pnas_mask(ax, annual_diff.T, annual_t_test_statistics.T, vmin, vmax, '(Hybrid minus SPEEDY)', cmap)

            fig.colorbar(scalarmap, ax=ax)
            fig.suptitle(f'{info[0]} [{info[1]}]\nAnnual Field Difference - {pressure_levels[i]}hPa')

            # plt.savefig(
            #     os.path.join(output_path, f'{var}_annual_field_diff_{pressure_levels[i]}hPa.png')
            # )
            plt.show()
            plt.close()


            ####### Plot annual & seasonal together
            # wind speed u has max of 6 and min of -4.5
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3, 
                ncols=1, 
                figsize=(8, 10),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            DJF_diff = DJF['hybrid'][..., i] - DJF['speedy'][..., i]
            DJF_s_1 = DJF['speedy_var'][..., i]
            DJF_s_2 = DJF['hybrid_var'][..., i]
            
            JJA_diff = JJA['hybrid'][..., i] - JJA['speedy'][..., i]
            JJA_s_1 = JJA['speedy_var'][..., i]
            JJA_s_2 = JJA['hybrid_var'][..., i]


            #Welch's t-test - for different variances
            DJF_t_test_statistics = DJF_diff/np.sqrt((DJF_s_1 + DJF_s_2)/seasons['DJF'])
            JJA_t_test_statistics = JJA_diff/np.sqrt((JJA_s_1 + JJA_s_2)/seasons['JJA'])

            vmin = min(np.min(annual_diff), np.min(DJF_diff), np.min(JJA_diff))
            vmax = max(np.max(annual_diff), np.max(DJF_diff), np.max(JJA_diff))

            plot_map_pnas_mask(ax1, annual_diff.T, annual_t_test_statistics.T, vmin, vmax, 'Annual', cmap)
            plot_map_pnas_mask(ax2, DJF_diff.T, DJF_t_test_statistics.T, vmin, vmax, 'DJF', cmap)
            scalarmap = plot_map_pnas_mask(ax3, JJA_diff.T, JJA_t_test_statistics.T, vmin, vmax, 'JJA', cmap)

            fig.colorbar(scalarmap, ax=(ax1, ax2, ax3))
            fig.suptitle(f'{info[0]} [{info[1]}]\nSeasonal Field Differences - {pressure_levels[i]}hPa\n(Hybrid minus SPEEDY)')

            # plt.savefig(
            #     os.path.join(output_path, f'{var}_seasonal_field_diff_{pressure_levels[i]}hPa.png')
            # )
            plt.show()
            plt.close()

    else:
        ####### Plot annual figure only
        fig, ax = plt.subplots(
            nrows=1, 
            ncols=1, 
            figsize=(8, 4),
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
        )

        annual_diff = annual['hybrid'] - annual['speedy']
        annual_s_1 = annual['speedy_var']
        annual_s_2 = annual['hybrid_var']

        #Welch's t-test - for different variances
        annual_t_test_statistics = annual_diff/np.sqrt((annual_s_1 + annual_s_2)/seasons['annual'])

        vmin = np.min(annual_diff)
        vmax = np.max(annual_diff)
        scalarmap = plot_map_pnas_mask(ax, annual_diff.T, annual_t_test_statistics.T, vmin, vmax, '(Hybrid minus SPEEDY)', cmap)

        fig.colorbar(scalarmap, ax=ax)
        fig.suptitle(f'{info[0]} [{info[1]}]\nAnnual Field Difference')

        # plt.savefig(
        #     os.path.join(output_path, f'{var}_annual_field_diff.png')
        # )
        plt.show()
        plt.close()


        ####### Plot annual & seasonal together
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, 
            ncols=1, 
            figsize=(8, 10),
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
        )

        DJF_diff = DJF['hybrid'] - DJF['speedy']
        DJF_s_1 = DJF['speedy_var']
        DJF_s_2 = DJF['hybrid_var']
        
        JJA_diff = JJA['hybrid'] - JJA['speedy']
        JJA_s_1 = JJA['speedy_var']
        JJA_s_2 = JJA['hybrid_var']


        #Welch's t-test - for different variances
        DJF_t_test_statistics = DJF_diff/np.sqrt((DJF_s_1 + DJF_s_2)/seasons['DJF'])
        JJA_t_test_statistics = JJA_diff/np.sqrt((JJA_s_1 + JJA_s_2)/seasons['JJA'])

        vmin = min(np.min(annual_diff), np.min(DJF_diff), np.min(JJA_diff))
        vmax = max(np.max(annual_diff), np.max(DJF_diff), np.max(JJA_diff))

        plot_map_pnas_mask(ax1, annual_diff.T, annual_t_test_statistics.T, vmin, vmax, 'Annual', cmap)
        plot_map_pnas_mask(ax2, DJF_diff.T, DJF_t_test_statistics.T, vmin, vmax, 'DJF', cmap)
        scalarmap = plot_map_pnas_mask(ax3, JJA_diff.T, JJA_t_test_statistics.T, vmin, vmax, 'JJA', cmap)

        fig.colorbar(scalarmap, ax=(ax1, ax2, ax3))
        fig.suptitle(f'{info[0]} [{info[1]}]\nSeasonal Field Differences\n(Hybrid minus SPEEDY)')

        # plt.savefig(
        #     os.path.join(output_path, f'{var}_seasonal_field_diff.png')
        # )
        plt.show()
        plt.close()