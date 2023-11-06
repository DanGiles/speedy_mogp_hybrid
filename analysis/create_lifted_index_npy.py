# Run this script AFTER running create_npy.py

import os
import numpy as np
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import parcel_profile
from metpy.calc import lifted_index
from metpy.units import units

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out

level_925 = 0
level_850 = 1
level_700 = 2
level_500 = 3

seasons = ['DJF', 'JJA']
locations = ['africa', 'india']

season_dump_counts = {'DJF': 3608, 'JJA': 3680}

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



for season in seasons:
    print(season)
    for location in locations:
        print(location)
        ############### SPEEDY ###############
        #### LOAD DATA
        if SPEEDY:
            print("SPEEDY")

            t_speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f'{location}_t_{season}.npy'))    # temperature (Kelvin) - (24, 8, 3608)
            q_speedy = np.load(os.path.join(analysis_root, 'SPEEDY', f'{location}_q_{season}.npy'))    # specific humidity (kg/kg) - (24, 8, 3608)

            n = q_speedy.shape[0]
            output_array = np.zeros((n, season_dump_counts[season])) + 9999

            for point in range(n):
                print(point)
                LI_SPEEDY = get_LI(t_speedy, q_speedy, point)
                output_array[point, 0:len(LI_SPEEDY)] = LI_SPEEDY
            
            np.save(
                os.path.join(analysis_root, "SPEEDY", f'{location}_lifted_index_{season}.npy'),
                output_array
            )


        ############### HYBRID ###############
        #### LOAD DATA
        if HYBRID:
            print(GP_name)

            t_hybrid = np.load(os.path.join(analysis_root, GP_name, f'{location}_t_{season}.npy'))    # temperature (Kelvin) - (24, 8, 3608)
            q_hybrid = np.load(os.path.join(analysis_root, GP_name, f'{location}_q_{season}.npy'))    # specific humidity (kg/kg) - (24, 8, 3608)

            n = q_hybrid.shape[0]
            output_array = np.zeros((n, season_dump_counts[season])) + 9999

            for point in range(n):
                print(point)
                LI_HYBRID = get_LI(t_hybrid, q_hybrid, point)
                output_array[point, 0:len(LI_HYBRID)] = LI_HYBRID

            np.save(
                os.path.join(analysis_root, GP_name, f'{location}_lifted_index_{season}.npy'),
                output_array
            )