# This script is VERY slow.
# It takes about 30 hours to complete. This should probably be parallelized.
# 200 chunks consumes about 8GB of RAM. Works well on my machine with 16GB RAM

import os
import numpy as np
import xarray as xr
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import parcel_profile
from metpy.calc import lifted_index
from metpy.units import units

from script_variables import *

# SPEEDY_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy' #override for local compute, otherwise comment out
# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out

level_925 = 0
level_850 = 1
level_700 = 2
level_500 = 3


def get_LI(t0, t1, t2, t3, q0, q1, q2, q3) -> np.ndarray:
    """Calculates and returns the array of lifted index values"""

    #### Find entries the q values are non-negative
    id_q_500_pos = q3[:] >= 0
    id_q_700_pos = q2[:] >= 0
    id_q_850_pos = q1[:] >= 0
    id_q_925_pos = q0[:] >= 0
    id_q_pos = id_q_500_pos & id_q_700_pos & id_q_850_pos & id_q_925_pos

    LI = np.zeros((sum(id_q_pos)))
    j=0

    #### DEWPOINT
    dewpoint = dewpoint_from_specific_humidity(
                (925)*units.hPa,
                t0*units.kelvin,
                q0*units('kg/kg')
            )
    for i in range(len(id_q_pos)):
        if id_q_pos[i] is np.True_:
            #### PARCEL PROFILE
            parcel_prof = np.array(parcel_profile(
                [925, 850, 700, 500]*units.hPa,
                t0[i]*units.kelvin,
                dewpoint[i].to(units.kelvin)
            ))
            # print(parcel_prof[0]) #Array of 4 values with units, but in Kelvin scale

            #### LIFTED INDEX
            LI[j] = np.array(lifted_index(
                [925, 850, 700, 500]*units.hPa,
                [t0[i], t1[i], t2[i], t3[i]]*units.kelvin,
                (parcel_prof*units.kelvin)
            )[0])
            j += 1
    return LI

def read_const_grd(filename, nlon, nlat, var):
    # Orography = 0
    # Land/Sea Mask = 1
    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]


output_dir = os.path.join(analysis_root, 'annual')

runs = ["HYBRID", "SPEEDY"]
nlon = 96
nlat = 48

lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
lsm = np.flip(lsm.T, 0) # lsm.shape = (48, 96)
poi = np.dstack(
    np.meshgrid(
        np.arange(0, nlon),
        np.arange(0, nlat)
    )
).reshape(-1, 2)
poi = poi[lsm.flatten() > 0.3, :] # Only keep cells with at least 30% land

for run in runs:
    # Loop through each location
    T = xr.load_dataset(os.path.join(output_dir, f'{run}_T.nc'), chunks={"timestamp": 200})
    Q = xr.load_dataset(os.path.join(output_dir, f'{run}_Q.nc'), chunks={"timestamp": 200})
    # ps = xr.load_dataset(os.path.join(output_dir, f'{run}_ps.nc'), chunks={"timestamp": 200})
    
    for i, (lon, lat) in enumerate(poi):
        # lat, lon = loc[0], loc[1]
        print(f'{run}: {i+1}/{len(poi)} - longitude: {lon}, latitude: {lat}')

        # Read in the NetCDFs
        T_fixed = T.isel(longitude=lon, latitude=lat)
        Q_fixed = Q.isel(longitude=lon, latitude=lat)
        # ps_fixed = ps.isel(longitude=lon, latitude=lat)

        # ps_0 = ps_fixed['ps'].data
        T_0 = T_fixed['T_0'].data
        T_1 = T_fixed['T_1'].data
        T_2 = T_fixed['T_2'].data
        T_3 = T_fixed['T_3'].data

        Q_0 = Q_fixed['Q_0'].data
        Q_1 = Q_fixed['Q_1'].data
        Q_2 = Q_fixed['Q_2'].data
        Q_3 = Q_fixed['Q_3'].data

        # output_array = np.zeros(ps_0.shape[0]) + 9999
        try:
            LI_output = get_LI(
                # ps_0, 
                T_0, T_1, T_2, T_3,
                Q_0, Q_1, Q_2, Q_3
            )
            # output_array[0:len(LI_output)] = LI_output
            np.save(os.path.join(output_dir, 'global_lifted_index', f'{run}_LI_{lon}_{lat}.npy'), LI_output)
        except:
            print(f'Error: {run} - lon: {lon}, lat: {lat}')
            continue