import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr


hybrid_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/myriad/run_1"
speedy_path = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs/neutral/speedy_myriad/annual"
ERA5_path = "/Users/dangiles/Documents/Stats/MetOffice/weather_data_processing/weatherbench"

# # Process the data
# hybrid = xr.load_dataset(os.path.join(hybrid_path, 'HYBRID_T.nc'))
# hybrid = hybrid.rename({'timestamp': 'time'})
# hybrid = hybrid.transpose('time', 'latitude', 'longitude')

# speedy = xr.load_dataset(os.path.join(speedy_path, f'SPEEDY_T.nc'))
# speedy = speedy.rename({'timestamp': 'time'})
# speedy = speedy.transpose('time', 'latitude', 'longitude')


# era5 = xr.open_dataset(os.path.join(ERA5_path, f"ERA_T.nc"))

# hybrid = hybrid.assign_coords(time=era5.time)
# speedy = speedy.assign_coords(time=era5.time)

# # 2. Create a list of all Q variables
# vars = [f'T_{i}' for i in range(8)] 

# # 3. Create a new dataset to store the results
# ds_monthly_speedy = xr.Dataset()
# ds_monthly_hybrid = xr.Dataset()

# # 4. Process each temperature variable
# for var in vars:
#     # Calculate the rolling monthly mean
#     rolling_mean_speedy = speedy[var].rolling(time=30*4, center=True, min_periods=1).mean()
#     rolling_mean_hybrid = hybrid[var].rolling(time=30*4, center=True, min_periods=1).mean()

#     # Resample to monthly frequency
#     monthly_mean_speedy = rolling_mean_speedy.resample(time='ME').mean()
#     monthly_mean_hybrid = rolling_mean_hybrid.resample(time='ME').mean()

#     # Add to the new dataset
#     ds_monthly_speedy[var] = monthly_mean_speedy
#     ds_monthly_hybrid[var] = monthly_mean_hybrid

# ds_monthly_speedy.to_netcdf(os.path.join(speedy_path,'T_monthly_means.nc'))
# ds_monthly_hybrid.to_netcdf(os.path.join(hybrid_path,'T_monthly_means.nc'))


ds_monthly_speedy = xr.load_dataset(os.path.join(speedy_path, 'T_monthly_means.nc'))
ds_monthly_hybrid = xr.load_dataset(os.path.join(hybrid_path, 'T_monthly_means.nc'))


ds_monthly_speedy = ds_monthly_speedy.mean('longitude')
ds_monthly_speedy = ds_monthly_speedy.mean('latitude')

ds_monthly_hybrid = ds_monthly_hybrid.mean('longitude')
ds_monthly_hybrid = ds_monthly_hybrid.mean('latitude')

# Create a list of markers for different levels
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
colors = plt.cm.rainbow(np.linspace(0, 1, len(ds_monthly_speedy.data_vars)))
sigma_levels = [0.95, 0.835, 0.685, 0.51, 0.34, 0.20, 0.095, 0.025]
pressure_levels = ['925', '850', '700', '500', '300', '200', '100', '30']

# Create the plot
plt.figure(figsize=(12, 6))

for i, (var, color) in enumerate(zip(ds_monthly_speedy.data_vars, colors)):
    
    if i == 0:
    # Plot the time series
        plt.plot(ds_monthly_speedy.time, 
                ds_monthly_speedy[var], 
                color = color,
                linestyle='-', 
                markersize=4,
                label=f'SPEEDY')
        
        plt.text(ds_monthly_speedy.time[-1]+2*(ds_monthly_speedy.time[-1]-ds_monthly_speedy.time[-2]), 
                ds_monthly_speedy[var][-1],
                f'{pressure_levels[i]}',
                color = color)
        
        plt.plot(ds_monthly_speedy.time, 
                ds_monthly_hybrid[var], 
                color = color,
                linestyle= '--',
                markersize=4, 
                label=f'Hybrid')
    elif i > 0 and i < len(ds_monthly_speedy.data_vars)-1:
        plt.plot(ds_monthly_speedy.time, 
                ds_monthly_speedy[var], 
                color = color,
                linestyle='-', 
                markersize=4)
        
        plt.text(ds_monthly_speedy.time[-1]+2*(ds_monthly_speedy.time[-1]-ds_monthly_speedy.time[-2]), 
                ds_monthly_speedy[var][-1],
                f'{pressure_levels[i]}',
                color = color)
        
        plt.plot(ds_monthly_speedy.time, 
                ds_monthly_hybrid[var], 
                color = color,
                linestyle= '--',
                markersize=4)
    else:
        plt.plot(ds_monthly_speedy.time, 
                ds_monthly_speedy[var], 
                color = color,
                linestyle='-', 
                markersize=4)
        
        plt.text(ds_monthly_speedy.time[-1]+2*(ds_monthly_speedy.time[-1]-ds_monthly_speedy.time[-2]), 
                ds_monthly_speedy[var][-1]+1,
                f'{pressure_levels[i]}',
                color = color)
        
        plt.plot(ds_monthly_speedy.time, 
                ds_monthly_hybrid[var], 
                color = color,
                linestyle= '--',
                markersize=4)

# Customize the plot
# plt.title('Monthly mean specific humidity at different atmospheric levels')
plt.title('Monthly mean temperature at different atmospheric levels')

plt.xlabel('Time')
# plt.ylabel('Specific Humidity [kg/kg]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(hybrid_path, f"T_monthly_means.png"), dpi = 300, bbox_inches='tight')
plt.show()