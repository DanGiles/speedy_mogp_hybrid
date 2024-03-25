# Easiest to run wherever mogp_emulator 0.6.1 is installed and where the trained MOGP objects are

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime, timedelta
import shutil # for removing data from previous simulations
import mogp_emulator
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from script_variables import *

def check_total_water(Q, Qs, rho):
    num = 0
    for i in range(len(Q[:,0])):
        water_content = np.sum(Q[i,:]*rho)
        sample = np.sum(Qs[i,:]*rho)
        diff = abs(sample - water_content)
        if diff > 1e-3:
            Qs[i,:] = Q[i,:]
            num +=1
    print("Number of physically inconsistent profiles (total water content) %i"%num)

    return Qs


def check_static_energy(Q, Qs, T, Ts):
    num = 0
    Cp = 1.005
    Lv = 2260
    for i in range(len(Q[:,0])):
        static_energy = np.sum(Q[i,:]*Lv + Cp*T[i,:])
        sample_static_energy = np.sum(Qs[i,:]*Lv + Cp*Ts[i,:])
        diff = abs(sample_static_energy - static_energy)
        if diff > 1.0:
            Ts[i,:] = T[i,:]
            num +=1
    print("Number of physically inconsistent profiles (moist static energy) %i"%num)

    return Ts


def make_dir(path: str) -> None:
    #do not empty directory if it doesn't exist!
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)
    # make directory
    os.mkdir(path)
    

def create_folders(output_folder):
    tmp = os.path.join(output_folder, "tmp")

    #do not empty directory if it doesn't exist!
    if os.path.isdir(tmp):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    os.mkdir(tmp)


def read_grd(filename, nlon, nlat, nlev):
    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data


def read_const_grd(filename, nlon, nlat, var):
    # Orography = 0
    # Land/Sea Mask = 1
    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]


def write_fortran(filename, data):
    f=open(filename,'wb+')
    # data = data.flatten()
    data = data.astype(np.float32)
    fortran_data=np.asfortranarray(data,'float32')
    fortran_data.T.tofile(f)
    f.close()
    return


def step_datetime(idate, dtdate, SPEEDY_DATE_FORMAT, dt):
    delta = timedelta(hours=dt)
    new_idate = datetime.strptime(idate, SPEEDY_DATE_FORMAT) + delta
    new_dtdate = datetime.strptime(dtdate, SPEEDY_DATE_FORMAT) + delta
    return new_idate.strftime(SPEEDY_DATE_FORMAT),new_dtdate.strftime(SPEEDY_DATE_FORMAT)


def read_oro_var() -> np.ndarray:
    oro_var_data = np.zeros((96, 48))
    oro_var_data_file = os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "std_orog_for_speedy.dat")
    with open(oro_var_data_file) as f:
        for row_i in range(96):
            oro_var_data[row_i, :] = np.fromstring(f.readline().strip(), dtype=float, sep=',')
    return oro_var_data


def data_prep(data, oro, ls, nlon, nlat) -> np.ndarray:
    T_mean = data[:,:,16:24]
    Q_mean = data[:,:,24:32]
    Q_mean = np.flip(Q_mean, axis = 2)
    T_mean = np.flip(T_mean, axis = 2)
    low_values_flags = Q_mean[:,:] < 1e-6  # Where values are low
    Q_mean[low_values_flags] = 1e-6

    if GP_name == "gp_without_oro_var":
        # Version for gp_without_oro_var
        train = np.empty(((nlon*nlat),19), dtype = np.float64)
        train[:, 0] = data[:,:,32].flatten()
        train[:, 1] = oro.flatten()
        train[:, 2] = ls.flatten()
        train[:, 3:11] = np.reshape(T_mean, ((nlon*nlat), 8))
        train[:, 11:] = np.reshape(Q_mean, ((nlon*nlat), 8))
    elif GP_name == "gp_with_oro_var":
        # Version for gp_with_oro_var
        train = np.empty(((nlon*nlat),20), dtype = np.float64)
        train[:, 0] = data[:,:,32].flatten()
        train[:, 1] = oro[...,0].flatten()
        train[:, 2] = oro[...,1].flatten() 
        train[:, 3] = ls.flatten()
        train[:, 4:12] = np.reshape(T_mean, ((nlon*nlat), 8))
        train[:, 12:] = np.reshape(Q_mean, ((nlon*nlat), 8))
    else:
        raise ValueError(f"GP_name not recognised, {GP_name} provided.")
    return train


def mogp_prediction_conserving(test, trained_gp, nlon, nlat, nlev, rho):
    variance, uncer, d = trained_gp.predict(test)
    print("Prediction")
    if GP_name == "gp_without_oro_var":
        T_mean = test[:, 3:11]
        Q_mean = test[:, 11:]
    elif GP_name == "gp_with_oro_var":
        T_mean = test[:, 4:12]
        Q_mean = test[:, 12:]
    resampled_T = np.empty((nlon*nlat*nlev), dtype = np.float64)
    resampled_Q = np.empty((nlon*nlat*nlev), dtype = np.float64)
    
    low_values_flags = variance < 1e-6  # Where values are low
    variance[low_values_flags] = 0.0

    resampled_T = np.random.normal(T_mean.flatten(), variance[:8,:].T.flatten())
    resampled_Q = np.random.normal(Q_mean.flatten(), variance[8:,:].T.flatten())

    resampled_Q = np.reshape(resampled_Q.T, (nlon*nlat, nlev))
    resampled_T = np.reshape(resampled_T.T, (nlon*nlat, nlev))

    resampled_Q = check_total_water(Q_mean, resampled_Q, rho)
    resampled_T = check_static_energy(Q_mean, resampled_Q, T_mean, resampled_T)

    resampled_T = np.reshape(resampled_T, (nlon, nlat, nlev))
    resampled_T  = np.flip(resampled_T, axis = 2)
    resampled_Q = np.reshape(resampled_Q, (nlon, nlat, nlev))
    resampled_Q  = np.flip(resampled_Q, axis = 2)

    return resampled_T, resampled_Q



def create_datasets():
    trained_gp = pickle.load(open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "rb"))

    # Defining constants and initial values
    SPEEDY_DATE_FORMAT = "%Y%m%d%H"

    IDate = "1982010100"
    dtDate = "1982010106"
    number_time_steps = (3652*4) 
    nlon = 96
    nlat = 48
    nlev = 8
    dt = 6

    n_JJA = 3680
    n_DJF = 3608
    counter_JJA = 0
    counter_DJF = 0
    
    # Initialisation steps
    output_folder = os.path.join(analysis_root, GP_name)
    data_folder = os.path.join(HYBRID_data_root, GP_name)
    nature_dir = os.path.join(SPEEDY_root, "DATA", "nature")

    data = read_grd(os.path.join(nature_dir, IDate +".grd"), nlon, nlat, nlev)

    # Read in the orography and land/sea fraction
    oro = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 0)
    lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
    oro = np.flip(oro, 1)
    lsm = np.flip(lsm, 1)
    # rho = np.loadtxt(os.path.join(HYBRID_root, "src", "density.txt"))
    if GP_name == "gp_with_oro_var":
        oro = np.stack((oro, read_oro_var()), axis=2)

    # Output Arrays
    T_var_JJA = np.empty((nlon, nlat, nlev, n_JJA))
    T_unc_JJA = np.empty_like(T_var_JJA)
    T_var_DJF = np.empty((nlon, nlat, nlev, n_DJF))
    T_unc_DJF = np.empty_like(T_var_DJF)
    Q_var_JJA = np.empty((nlon, nlat, nlev, n_JJA))
    Q_unc_JJA = np.empty_like(Q_var_JJA)
    Q_var_DJF = np.empty((nlon, nlat, nlev, n_DJF))
    Q_unc_DJF = np.empty_like(Q_var_DJF)

    # Main time loop
    for t in range(0,number_time_steps):
        # Time counters
        print(IDate, dtDate, t)

        if int(IDate[4:6]) in [12, 1, 2, 6, 7, 8]:
            if int(IDate[4:6]) in [12, 1, 2]:
                winter = True
            else:
                winter = False

            test = data_prep(data, oro, lsm, nlon, nlat)

            variance, uncer, d = trained_gp.predict(test)

            T_var = variance[:8,:].T
            Q_var = variance[8:,:].T
            T_unc = uncer[:8,:].T
            Q_unc = uncer[8:,:].T

            if winter:
                T_var_DJF[..., counter_DJF] = np.reshape(T_var, (nlon, nlat, nlev))
                T_unc_DJF[..., counter_DJF] = np.reshape(T_unc, (nlon, nlat, nlev))
                Q_var_DJF[..., counter_DJF] = np.reshape(Q_var, (nlon, nlat, nlev))
                Q_unc_DJF[..., counter_DJF] = np.reshape(Q_unc, (nlon, nlat, nlev))
                counter_DJF += 1
            else:
                T_var_JJA[..., counter_JJA] = np.reshape(T_var, (nlon, nlat, nlev))
                T_unc_JJA[..., counter_JJA] = np.reshape(T_unc, (nlon, nlat, nlev))
                Q_var_JJA[..., counter_JJA] = np.reshape(Q_var, (nlon, nlat, nlev))
                Q_unc_JJA[..., counter_JJA] = np.reshape(Q_unc, (nlon, nlat, nlev))
                counter_JJA += 1

        # # # Read Speedy output
        file = os.path.join(data_folder, (dtDate+".grd"))
        data = read_grd(file, nlon, nlat, nlev)
        # # Update time counters
        IDate, dtDate = step_datetime(IDate, dtDate, SPEEDY_DATE_FORMAT, dt)

    np.save(os.path.join(output_folder, "prediction_T_var_DJF.npy"), T_var_DJF)
    np.save(os.path.join(output_folder, "prediction_T_unc_DJF.npy"), T_unc_DJF)
    np.save(os.path.join(output_folder, "prediction_Q_var_DJF.npy"), Q_var_DJF)
    np.save(os.path.join(output_folder, "prediction_Q_unc_DJF.npy"), Q_unc_DJF)

    np.save(os.path.join(output_folder, "prediction_T_var_JJA.npy"), T_var_JJA)
    np.save(os.path.join(output_folder, "prediction_T_unc_JJA.npy"), T_unc_JJA)
    np.save(os.path.join(output_folder, "prediction_Q_var_JJA.npy"), Q_var_JJA)
    np.save(os.path.join(output_folder, "prediction_Q_unc_JJA.npy"), Q_unc_JJA)


def main():
    # optional routine. This should only be run once for each GP
    create_datasets()

    def plot_map(ax, field_data, title, heatmap=None, **kwargs):
        ax.coastlines()
        heatmap = ax.contourf(lon_grid, lat_grid, field_data, **kwargs)
        # heatmap.set_clim(**kwargs)
        ax.set_title(title)
        return heatmap

    # create plots
    nlon = 96
    nlat = 48
    nlev = 8

    # Set up the coordinate system
    lon = np.linspace(-180, 180, nlon, endpoint=False) # endpoint=False to match SPEEDY
    # lat = np.linspace(-90, 90, nlat) # this does NOT match SPEEDY
    lat_vals = "-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989  -9.278  -5.567  -1.856   1.856   5.567   9.278  12.989  16.700  20.411  24.122  27.833  31.545  35.256  38.967  42.678  46.389  50.099  53.810  57.521  61.232  64.942  68.652  72.362  76.070  79.777  83.479  87.159"
    lat = np.array([float(val) for val in lat_vals.split()]) # to match SPEEDY
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    data_folder = os.path.join(analysis_root, GP_name)
    output_path = os.path.join(pngs_root, GP_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, 'field_predictions')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    pressure_levels = [30, 100, 200, 300, 500, 700, 850, 925] # hPa
    variables = ['T', 'Q']
    seasons = ['DJF', 'JJA']
    
    for variable in variables:
        for season in seasons:
            var = np.load(os.path.join(
                data_folder, 
                f"prediction_{variable}_var_{season}.npy"
            ))
            unc = np.load(os.path.join(
                data_folder, 
                f"prediction_{variable}_unc_{season}.npy"
            ))
            
            for l, pressure_level in enumerate(pressure_levels):
                fig, axes = plt.subplots(
                    nrows=2, 
                    ncols=1,
                    figsize=(8, 8),
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
                )

                heatmap1 = plot_map(
                    axes[0], 
                    np.mean(var[..., l], axis=-1).T, 
                    "GP Mean"
                )
                heatmap2 = plot_map(
                    axes[1], 
                    np.mean(unc[..., l], axis=-1).T, 
                    "GP Variance"
                )

                plt.colorbar(heatmap1, ax=axes[0])
                plt.colorbar(heatmap2, ax=axes[1])
                fig.suptitle(f'MOGP Predictions - {pressure_level}hPa')
                plt.savefig(
                    os.path.join(
                        output_path, 
                        f'gp_pred_{variable}_10year_{season}_{pressure_level}hPa.png'
                    ),
                    dpi=200
                )
                plt.close()

    return


if __name__ == '__main__':
    main()