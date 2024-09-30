import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr


def interpolate_to_speedy(ds, variable_name):
    # # Define the new latitude and longitude coordinates
    min_lat, max_lat = ds[variable_name].latitude.min(), ds[variable_name].latitude.max()
    min_lon, max_lon = ds[variable_name].longitude.min(), ds[variable_name].longitude.max()
    num_lats, num_lons = 48, 96  # Adjust as needed
    print(min_lon, max_lon, min_lat, max_lat)
    # new_lats = np.linspace(min_lat, max_lat, num_lats)
    lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
    new_lats = np.array([float(val) for val in lat_vals.split()])
    new_lons = np.linspace(min_lon, max_lon, num_lons, endpoint=True)

    # Create a new grid
    new_grid = xr.Dataset({'latitude': (['latitude'], new_lats),
                        'longitude': (['longitude'], new_lons)})
    
    field = ds[variable_name]

    interp = field.interp(latitude=new_grid.latitude, longitude=new_grid.longitude, method='linear')

    return interp

hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/run_1"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy_myriad/annual"
GPCP_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/gpcp"
ERA5_path = "/Users/dangiles/Documents/Stats/MetOffice/weather_data_processing/weatherbench"


# # Process the data
hybrid = xr.load_dataset(os.path.join(hybrid_path, 'HYBRID_precip.nc'))
hybrid = hybrid.rename({'timestamp': 'time'})
hybrid = hybrid.transpose('time', 'latitude', 'longitude')

speedy = xr.load_dataset(os.path.join(speedy_path, f'SPEEDY_precip.nc'))
speedy = speedy.rename({'timestamp': 'time'})
speedy = speedy.transpose('time', 'latitude', 'longitude')

era5 = xr.open_dataset(os.path.join(ERA5_path, f"ERA_T.nc"))


hybrid = hybrid.assign_coords(time=era5.time)
speedy = speedy.assign_coords(time=era5.time)


# 3. Create a new dataset to store the results
ds_monthly_speedy = xr.Dataset()
ds_monthly_hybrid = xr.Dataset()


# Calculate the rolling monthly mean
rolling_mean_speedy = speedy['precip'].rolling(time=30*4, center=True, min_periods=1).mean()
rolling_mean_hybrid = hybrid['precip'].rolling(time=30*4, center=True, min_periods=1).mean()

# Resample to monthly frequency
monthly_mean_speedy = rolling_mean_speedy.resample(time='ME').mean()
monthly_mean_hybrid = rolling_mean_hybrid.resample(time='ME').mean()

# Add to the new dataset
ds_monthly_speedy['precip'] = monthly_mean_speedy
ds_monthly_hybrid['precip'] = monthly_mean_hybrid

ds_monthly_speedy.to_netcdf(os.path.join(speedy_path,'Precip_monthly_means.nc'))
ds_monthly_hybrid.to_netcdf(os.path.join(hybrid_path,'Precip_monthly_means.nc'))


ds_monthly_speedy = xr.load_dataset(os.path.join(speedy_path, 'Precip_monthly_means.nc'))
ds_monthly_hybrid = xr.load_dataset(os.path.join(hybrid_path, 'Precip_monthly_means.nc'))
ds_monthly_gpcp = xr.load_dataset(os.path.join(GPCP_path, 'precipitation.nc'))


print(ds_monthly_gpcp)
ds_monthly_gpcp = interpolate_to_speedy(ds_monthly_gpcp, 'precip')
print(ds_monthly_gpcp)

ds_monthly_speedy_var = ds_monthly_speedy.var('time')
ds_monthly_hybrid_var = ds_monthly_hybrid.var('time')
ds_monthly_gpcp_var = ds_monthly_gpcp.var('time')

ds_monthly_speedy_var.to_netcdf(os.path.join(speedy_path,'Precip_monthly_var.nc'))
ds_monthly_hybrid_var.to_netcdf(os.path.join(hybrid_path,'Precip_monthly_var.nc'))
ds_monthly_gpcp_var.to_netcdf(os.path.join(GPCP_path,'Precip_monthly_var.nc'))

plt.figure(1)
plt.contourf(ds_monthly_speedy_var['precip'])
plt.title('Speedy')

plt.figure(2)
plt.contourf(ds_monthly_speedy_var['precip'])
plt.title('Hybrid')

plt.figure(3)
plt.contourf(ds_monthly_speedy_var['precip'])
plt.title('GPCP')
plt.show()
