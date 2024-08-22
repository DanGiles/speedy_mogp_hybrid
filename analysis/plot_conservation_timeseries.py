import os
import numpy as np
# from datetime import datetime, date, timedelta
# import shutil # for removing data from previous simulations
# import pickle
# import sys

import xarray as xr # groupBy operations are faster if package flox is installed

import matplotlib.pyplot as plt

from script_variables import *

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
for run in ['SPEEDY', 'HYBRID']:
# for run in ['HYBRID']:
    print(run)
    water_and_energy = xr.open_dataset(f'data/analysis/annual/{run}_water_and_energy.nc', engine='netcdf4', chunks={'timestamp': 500})
    
    # Fix the longitude to a single point 90. 270 would also work well?
    lon = 90
    # lats = np.linspace(-80, 70, 5, endpoint=False)
    lats = [25]

    w_e_lon = water_and_energy.sel(longitude=lon, method='nearest')
    # print(w_e_lon, '\n')

    # constrain to a few latitudes
    for lat in lats:
        # The plan was to plot several locations on the same graph, but the data is too noisy and hard to read.
        w_e_lon_lat = w_e_lon.sel(latitude=lat, method='nearest')
        ax[0].plot(w_e_lon_lat['timestamp'], w_e_lon_lat['total_water_content'], label=f'{run} {lat}° lat')
        ax[1].plot(w_e_lon_lat['timestamp'], w_e_lon_lat['static_energy'], label=f'{run} {lat}° lat')

ax[0].set_title(f'Total water content - {lon}° longitude')
ax[1].set_title(f'Static energy - {lon}° longitude')
ax[0].set_xlabel('Time')
ax[1].set_xlabel('Time')
ax[0].set_ylabel('Total water content') # [kg/m^2]???
ax[1].set_ylabel('Static energy') # [J/kg]???
ax[0].legend()
ax[1].legend()
plt.tight_layout()
# plt.legend()
plt.show()