import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from script_variables import *

SPEEDY_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy' #override for local compute, otherwise comment out
analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out


def round_nearest_half(x):
    return round(x * 2.0)/2

nlon = 96
nlat = 48
nlev = 8

# Set up the coordinate system
lons = np.linspace(0, 360, nlon, endpoint=False) # endpoint=False to match SPEEDY
# lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
lats = np.array([float(val) for val in lat_vals.split()])

##### Loop through all the files #####
speedy_m2 = np.zeros((nlon, nlat))
hybrid_m2 = np.zeros((nlon, nlat))
speedy_m4 = np.zeros((nlon, nlat))
hybrid_m4 = np.zeros((nlon, nlat))
speedy_n = np.zeros((nlon, nlat))
hybrid_n = np.zeros((nlon, nlat))
speedy_mean = np.zeros((nlon, nlat))
hybrid_mean = np.zeros((nlon, nlat))

folder = os.path.join(analysis_root, 'annual', 'global_lifted_index')
for filename in os.listdir(folder):
    if filename.startswith('SPEEDY_LI'):
        lon = int(filename.split('_')[2])
        lat = int(filename.split('_')[3].split('.')[0])
        LI = np.load(os.path.join(folder, filename))
        speedy_m2[lon, lat] = np.sum(LI < -2)
        speedy_m4[lon, lat] = np.sum(LI < -4)
        speedy_n[lon, lat] = len(LI)
        speedy_mean[lon, lat] = np.mean(LI)
    elif filename.startswith('HYBRID_LI'):
        lon = int(filename.split('_')[2])
        lat = int(filename.split('_')[3].split('.')[0])
        LI = np.load(os.path.join(folder, filename))
        hybrid_m2[lon, lat] = np.sum(LI < -2)
        hybrid_m4[lon, lat] = np.sum(LI < -4)
        hybrid_n[lon, lat] = len(LI)
        hybrid_mean[lon, lat] = np.mean(LI)


def read_const_grd(filename, nlon, nlat, var):
    # Orography = 0
    # Land/Sea Mask = 1
    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]
lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1) # lsm.shape = (96, 48)
lsm = np.flip(lsm.T, 0) # lsm.shape = (48, 96)


def plot(ax, field_data, title):
    vmin = np.nanmin(field_data)
    vmax = np.nanmax(field_data)
    vabsmax = max(abs(vmin), abs(vmax))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vabsmax, vcenter=0, vmax=vabsmax)

    map = ax.pcolormesh(field_data, cmap='seismic', norm=norm)
    fig.colorbar(map, ax=ax)
    ax.set_title(f"Difference in {title}")
    ax.set_xticks([0, 24, 48, 72, 96])
    # ax.set_xticklabels([-180, -90, 0, 90, 180])
    ax.set_xticklabels([0, 90, 180, 270, 360])
    ax.set_yticks([0, 12, 24, 36, 48])
    ax.set_yticklabels([-90, -45, 0, 45, 90])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(15, 9),
    # subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
)
fig.suptitle('Lifted Index over land\nHybrid minus SPEEDY')
plot(
    axes.flatten()[0], 
    (hybrid_m2 - speedy_m2).T,
    f"#(LI<2)"
)
plot(
    axes.flatten()[1], 
    (hybrid_m2/hybrid_n - speedy_m2/speedy_n).T,
    f"% of points where (LI<2)"
)
plot(
    axes.flatten()[2], 
    (hybrid_m4 - speedy_m4).T,
    f"#(LI<4)"
)
plot(
    axes.flatten()[3], 
    (hybrid_m4/hybrid_n - speedy_m4/speedy_n).T,
    f"% of points where (LI<4)"
)
plt.savefig(os.path.join(pngs_root, 'annual', 'LI_diff.png'))
# plt.show()