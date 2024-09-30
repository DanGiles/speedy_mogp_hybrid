import numpy as np
import os
import xarray as xr
from pathlib import Path

def preprocess_weatherbench_T_q(var_name, output):
    """
    Preprocess the ERA5 dataset from WeatherBench

    there are 32 variables in the dataset but extract geopotential

    test data shape: (1460, 64, 32, 1)
    train data shape : (1460, 64, 32, 1)
    """
    GCP_PATH = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'

    # Load in the data
    ds = xr.open_zarr(GCP_PATH)
    target_start = np.datetime64('1982-01-01T00:00:00.000000000')
    target_end = np.datetime64('1992-01-01T00:00:00.000000000')


    # Extract the data
    var = ds[var_name].sel(time=slice(target_start, target_end))

    var = var.sel(level=925)
    print(var)
    chunk_size = 50
    ds.close()

    # Interpolate to Speedy Grid
    # # Define the new latitude and longitude coordinates
    min_lat, max_lat = var.latitude.min(), var.latitude.max()
    min_lon, max_lon = var.longitude.min(), var.longitude.max()
    num_lats, num_lons = 48, 96  # Adjust as needed
   
    # new_lats = np.linspace(min_lat, max_lat, num_lats)
    lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
    new_lats = np.array([float(val) for val in lat_vals.split()])
    new_lons = np.linspace(min_lon, max_lon, num_lons, endpoint=True)

    # Create a new grid
    new_grid = xr.Dataset({'latitude': (['latitude'], new_lats),
                        'longitude': (['longitude'], new_lons)})

    # Initialize an empty list to store the interpolated chunks
    interp_chunks = []

    num_time_chunks = len(var['time']) // chunk_size

    # Loop through the dataset in chunks along the time dimension
    for i in range(num_time_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_time_chunks - 1 else None
        print(start_idx, end_idx)
        # Read the field for the current time chunk
        field_chunk = var.isel(time=slice(start_idx, end_idx))

        # Interpolate the time chunk to the new grid
        interp_chunk = field_chunk.interp(latitude=new_grid.latitude, longitude=new_grid.longitude, method='linear')
        
        # Append the interpolated chunk to the list
        interp_chunks.append(interp_chunk)

    # # Concatenate the interpolated chunks into a single DataArray
    interp_field = xr.concat(interp_chunks, dim='time')

    # Create a new Dataset with the interpolated data
    output_ds = xr.Dataset({var_name: interp_field})

    # Write the output Dataset to a netCDF file
    output_ds.to_netcdf(f'{output}.nc', compute=False)




if __name__ == '__main__':
    #### WeatherBench datasets
    preprocess_weatherbench_T_q('temperature', 'ERA_T')
    preprocess_weatherbench_T_q('specific_humidity', 'ERA_Q')