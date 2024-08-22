########################### READ ME ###########################
# This script only uses the first 5 variables for both Q and T.
# This is inline with the current implementation in `wrapper.py`.
# Change lines 26, 40 and 41 if you want to use all variables.
###############################################################

import os
import numpy as np
# from datetime import datetime, date, timedelta
# import shutil # for removing data from previous simulations
# import pickle
# import sys

import xarray as xr # groupBy operations are faster if package flox is installed

from script_variables import *


def total_water(Q: xr.Dataset):
    """Check total water content of the model output

    :param Q: Dataset of specific humidity vertical profile

    :return: water_content: total water content of the vertical profile
    """
    rho = np.array([1.1379, 1.0626, 0.908 , 0.6914, 0.4572, 0, 0, 0])
    water_content = Q['Q_0']*rho[0] + Q['Q_1']*rho[1] + Q['Q_2']*rho[2] + Q['Q_3']*rho[3] + Q['Q_4']*rho[4]
    return water_content


def static_energy(QandT: xr.Dataset):
    """Check the static energy of the model output
    
    :param QandT: Dataset of specific humidity and temperature vertical profiles
    
    :return: static_energy: static energy of the vertical profile
    """
    Cp = 1.005
    Lv = 2260
    
    Q_sum = QandT['Q_0'] + QandT['Q_1'] + QandT['Q_2'] + QandT['Q_3'] + QandT['Q_4']
    T_sum = QandT['T_0'] + QandT['T_1'] + QandT['T_2'] + QandT['T_3'] + QandT['T_4']
    static_energy_val = Lv*Q_sum + Cp*T_sum

    return static_energy_val


for run in ['HYBRID', 'SPEEDY']:
    print(run)
    # Q = xr.open_dataset(f'data/analysis/annual/{run}_Q.nc', engine='netcdf4', chunks='auto')
    # T = xr.open_dataset(f'data/analysis/annual/{run}_T.nc', engine='netcdf4', chunks='auto')
    Q = xr.open_dataset(f'data/analysis/annual/{run}_Q.nc', engine='netcdf4', chunks={'timestamp': 500})
    T = xr.open_dataset(f'data/analysis/annual/{run}_T.nc', engine='netcdf4', chunks={'timestamp': 500})

    # Merge datasets
    QandT = xr.merge([Q, T])
    print(QandT, '\n')

    # Apply function using Dask
    total_water_out = QandT.map_blocks(total_water)
    total_water_out.name = 'total_water_content'
    static_energy_out = QandT.map_blocks(static_energy)
    static_energy_out.name = 'static_energy'

    print(total_water_out, '\n')
    print(static_energy_out)

    water_and_energy = xr.merge([total_water_out, static_energy_out])
    water_and_energy.to_netcdf(f'data/analysis/annual/{run}_water_and_energy.nc', engine='netcdf4')