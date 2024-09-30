import xarray as xr
import glob
ds = xr.merge([xr.open_dataset(f) for f in glob.glob('gpcp_v02r03_monthly_d*')])

ds.to_netcdf('precipitation.nc')
ds = xr.open_dataset('precipitation.nc')

mean = ds.precip.mean('time')

variable_name = 'precip'
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

interp_mean = mean.interp(latitude=new_grid.latitude, longitude=new_grid.longitude, method='linear')

interp_mean.to_netcdf('precipitation_inter.nc')