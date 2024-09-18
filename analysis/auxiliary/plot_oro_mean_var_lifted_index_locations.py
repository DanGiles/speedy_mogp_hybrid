import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from typing import List


SPEEDY_root =    '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy/' #override for local compute, otherwise comment out

nlon = 96
nlat = 48
nlev = 8

oro_var_data_file = os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "std_orog_for_speedy.dat")

def read_oro_var() -> np.ndarray:
    oro_var_data = np.zeros((96, 48))
    with open(oro_var_data_file) as f:
        for row_i in range(96):
            oro_var_data[row_i, :] = np.fromstring(f.readline().strip(), dtype=float, sep=',')
    
    return oro_var_data


def read_const_grd(filename, nlon, nlat, var):
    # Orography = 0
    # Land/Sea Mask = 1
    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]


def plot_map(ax, field_data, title, heatmap=None, **kwargs):
    ax.coastlines()
    heatmap = ax.contourf(lon_grid, lat_grid, field_data.T, **kwargs)
    # heatmap.set_clim(**kwargs)
    ax.set_title(title)
    return heatmap


# Set up the coordinate system
lon = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
lon_grid, lat_grid = np.meshgrid(lon, lat)

oro = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 0)
oro = np.flip(oro, 1)




# fig, axes = plt.subplots(
#     nrows=2, 
#     ncols=1,
#     figsize=(8, 8),
#     subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
# )
# heatmap1 = plot_map(axes[0], oro, "oro mean")
# heatmap2 = plot_map(axes[1], read_oro_var(), "oro var")
# plt.colorbar(heatmap1, ax=axes[0])
# plt.colorbar(heatmap2, ax=axes[1])
# # plt.savefig(
# #     os.path.join('/Users/jamesbriant/Desktop', f'oro_mean_vs_var.png'),
# #     dpi=200
# # )
# plt.show()
# plt.close()




def plot_points(ax, lon_index_points: List[int], lat_index_points: List[int], col: str = 'orange'):
    n1 = len(lon_index_points)
    n2 = len(lat_index_points)

    lon_index = lon_index_points*n2

    lat_index = []
    [lat_index.extend([index]*n1) for index in lat_index_points]


    ax.scatter(
        lon[lon_index], 
        lat[lat_index], 
        c=col
    )

    for i in range(n1):
        ax.annotate(i+1, (lon[lon_index][i], lat[lat_index][i]))

    for i in range(1,n2):
        ax.annotate(i*n1+1, (lon[lon_index][i*n1], lat[lat_index][i*n1]))
        ax.annotate((i+1)*n1, (lon[lon_index][(i+1)*n1-1], lat[lat_index][(i+1)*n1-1]))




# Lifted index points
fig, ax = plt.subplots(
    nrows=1, 
    ncols=1,
    figsize=(16, 8),
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
)

heatmap1 = plot_map(ax, oro, "oro mean")
# plt.colorbar(heatmap1, ax=ax)
# plt.savefig(
#     os.path.join('/Users/jamesbriant/Desktop', f'oro_mean_lifted_index_points.png'),
#     dpi=200
# )


#################
##### INDIA #####
#################

lon_index_india_points = [17,18,19,20,21,22,23,24]
lat_index_india_points = [28,27,26,25,24,23]

plot_points(ax, lon_index_india_points, lat_index_india_points)


##################
##### AFRICA #####
##################

lon_index_arabia_points = [13, 14, 15, 16]
lat_index_arabia_points = [28, 27, 26, 25, 24, 23, 22]

plot_points(ax, lon_index_arabia_points, lat_index_arabia_points, 'green')


##################
##### AFRICA #####
##################

lon_index_africa_points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
lat_index_africa_points = [25,24,23,22]

plot_points(ax, lon_index_africa_points, lat_index_africa_points, 'red')

plt.show()
plt.close()