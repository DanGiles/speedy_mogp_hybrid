# Run this script AFTER running create_npy.py

import os
import numpy as np
import xarray as xr
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import parcel_profile
from metpy.calc import lifted_index
from metpy.units import units

from script_variables import *

output_dir = os.path.join(analysis_root, 'annual')

runs = ["HYBRID", "SPEEDY"]
nlon = 96
nlat = 48

# List of points of interest
poi = [(9, 23), (8, 24), (21, 26) , (21, 25) , (22, 25), 
       (28, 24), (29, 24), (30, 24), (70, 26), (71, 26), 
       (72, 26), (73, 26), (74, 26), (75, 26), (90, 23), 
       (93, 25)]

for run in runs:
    print(run)
    # Read in the NetCDFs
    precip = xr.load_dataset(os.path.join(output_dir, f'{run}_precip.nc'))

    precip = precip['precip'].data
    # Loop through each location
    for loc in poi:
        lon, lat = loc[0], loc[1]
        print(lon, lat)
        np.save(os.path.join(output_dir, 'global_precip', f'{run}_precip_{lon}_{lat}.npy'), precip[:,lon,lat])
