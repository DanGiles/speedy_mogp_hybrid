import os
import numpy as np
import xarray as xr
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import parcel_profile
from metpy.calc import lifted_index
from metpy.units import units


level_925 = 0
level_850 = 1
level_700 = 2
level_500 = 3


# pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500] # *units.Pa
# sigma_levels = [0.95, 0.835, 0.685, 0.51, 0.34, 0.20, 0.095, 0.025]

def get_LI(ps, t0, t1, t2, t3, q0, q1, q2, q3) -> np.ndarray:
    """Calculates and returns the array of lifted index values"""

    # press = [(sigma_levels[0]*ps*0.01), (sigma_levels[1]*ps*0.01), (sigma_levels[2]*ps*0.01), (sigma_levels[3]*ps*0.01)]
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


hybrid_path = "./run_1/annual"
speedy_path = "./speedy/annual"

runs = ["HYBRID", "SPEEDY"]
nlon = 96
nlat = 48

# List of points of interest
poi = [(9, 23), (8, 24), (21, 26) , (21, 25) , (22, 25), 
       (28, 24), (29, 24), (30, 24), (70, 26), (71, 26), 
       (72, 26), (73, 26), (74, 26), (75, 26), (90, 23), 
       (93, 25)]
# poi = [(8, 24)]

for run in runs:
    print(run)
    # Read in the NetCDFs
    if run == "HYBRID":
        T = xr.load_dataset(os.path.join(hybrid_path, f'{run}_T.nc'))
        Q = xr.load_dataset(os.path.join(hybrid_path, f'{run}_Q.nc'))
        ps = xr.load_dataset(os.path.join(hybrid_path, f'{run}_ps.nc'))
    else:
        T = xr.load_dataset(os.path.join(speedy_path, f'{run}_T.nc'))
        Q = xr.load_dataset(os.path.join(speedy_path, f'{run}_Q.nc'))
        ps = xr.load_dataset(os.path.join(speedy_path, f'{run}_ps.nc'))

    ps_0 = ps['ps'].data
    T_0 = T['T_0'].data
    T_1 = T['T_1'].data
    T_2 = T['T_2'].data
    T_3 = T['T_3'].data

    Q_0 = Q['Q_0'].data
    Q_1 = Q['Q_1'].data
    Q_2 = Q['Q_2'].data
    Q_3 = Q['Q_3'].data
    # Loop through each location
    for loc in poi:
        lon, lat = loc[0], loc[1]
        print(lon, lat)
        output_array = np.zeros(ps_0.shape[0]) + 9999
        LI_output = get_LI(ps_0[:, lon, lat], 
                T_0[:, lon, lat], 
                T_1[:, lon, lat], 
                T_2[:, lon, lat], 
                T_3[:, lon, lat],
                Q_0[:, lon, lat], 
                Q_1[:, lon, lat], 
                Q_2[:, lon, lat], 
                Q_3[:, lon, lat])
        output_array[0:len(LI_output)] = LI_output
        np.save(os.path.join(hybrid_path, 'global_lifted_index', f'{run}_LI_{lon}_{lat}.npy'), LI_output)