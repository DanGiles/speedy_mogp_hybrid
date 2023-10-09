import os
import numpy as np
import matplotlib.pyplot as plt
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import parcel_profile
from metpy.calc import lifted_index
from metpy.units import units

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

level_925 = 0
level_850 = 1
level_700 = 2
level_500 = 3

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'lifted_index')
if not os.path.isdir(output_path):
    os.mkdir(output_path)

seasons = ['DJF', 'JJA']
locations = ['africa', 'india']

# pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500] # *units.Pa


def get_LI(t, q, location) -> np.ndarray:
    """Calculates and returns the array of lifted index values"""

    #### Find entries the q values are non-negative
    id_q_500_pos = q[location, level_500, :] >= 0
    id_q_700_pos = q[location, level_700, :] >= 0
    id_q_850_pos = q[location, level_850, :] >= 0
    id_q_925_pos = q[location, level_925, :] >= 0
    id_q_pos = id_q_500_pos & id_q_700_pos & id_q_850_pos & id_q_925_pos

    LI = np.zeros((sum(id_q_pos)))
    j=0
    for i in range(len(id_q_pos)):
        if id_q_pos[i] is np.True_:
            #### DEWPOINT
            dewpoint = dewpoint_from_specific_humidity(
                925*units.hPa,
                t[location, level_925, i]*units.kelvin,
                q[location, level_925, i]*units('kg/kg')
            )
            # print(dewpoint) #A single value with units Celsius

            #### PARCEL PROFILE
            parcel_prof = np.array(parcel_profile(
                [925, 850, 700, 500]*units.hPa,
                t[location, level_925, i]*units.kelvin,
                dewpoint
            ))
            # print(parcel_prof[0]) #Array of 4 values with units, but in Kelvin scale

            #### LIFTED INDEX
            LI[j] = np.array(lifted_index(
                [925, 850, 700, 500]*units.hPa,
                t[location, [level_925, level_850, level_700, level_500], i]*units.kelvin,
                parcel_prof*units.kelvin
            )[0])
            j += 1
    return LI


def round_nearest_half(x):
    return round(x * 2.0)/2

for location in locations:
    for season in seasons:
        ############### SPEEDY ###############
        #### LOAD DATA
        t_speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f'{location}_t_{season}.npy'))    # temperature (Kelvin) - (24, 8, 3608)
        q_speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f'{location}_q_{season}.npy'))    # specific humidity (kg/kg) - (24, 8, 3608)

        ############### HYBRID ###############
        #### LOAD DATA
        t_hybrid = np.load(os.path.join(analysis_root, GP_name, f'{location}_t_{season}.npy'))    # temperature (Kelvin) - (24, 8, 3608)
        q_hybrid = np.load(os.path.join(analysis_root, GP_name, f'{location}_q_{season}.npy'))    # specific humidity (kg/kg) - (24, 8, 3608)

        for point in range(24):
            LI_SPEEDY = get_LI(t_speedy, q_speedy, point)
            LI_HYBRID = get_LI(t_hybrid, q_hybrid, point)

            bin_min = round_nearest_half(min(np.min(LI_SPEEDY), np.min(LI_HYBRID)))
            bin_max = round_nearest_half(max(np.max(LI_SPEEDY), np.max(LI_HYBRID)))
            bin_points = np.arange(bin_min, bin_max, 0.5)

            ############### PLOTTING ###############
            fig, ax = plt.subplots(
                1, 1, 
                figsize=(8, 8)
            )
            fig.suptitle(f'Lifted Index Histogram \n Season: {season} - Location: {location.capitalize()}, region {point+1}.')

            y1, x1, _ = ax.hist(LI_SPEEDY, bins=bin_points, alpha=0.5, label="SPEEDY")
            y2, x2, _ = ax.hist(LI_HYBRID, bins=bin_points, alpha=0.5, label="Hybrid")

            y = max(max(y1), max(y2)) + 50

            ax.vlines(np.mean(LI_SPEEDY), 0, y, colors='tab:blue', linestyles='dashed')
            ax.vlines(np.mean(LI_HYBRID), 0, y, colors='tab:orange', linestyles='dashed')

            ax.set_title(f'SPEEDY - mean: {np.mean(LI_SPEEDY):.3f}; var: {np.var(LI_SPEEDY):.3f} \n HYBRID - mean: {np.mean(LI_HYBRID):.3f}; var: {np.var(LI_HYBRID):.3f}')
            ax.legend()
            
            plt.savefig(
                os.path.join(output_path, f'{location}_{point+1}_{season}_lifted_index.png')
            )
            plt.close()