# Easiest to run wherever mogp_emulator 0.6.1 is installed and where the trained MOGP objects are

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import mogp_emulator
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from script_variables import *

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'profiles')
if not os.path.isdir(output_path):
    os.mkdir(output_path)


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
    elif GP_name == "gp_with_oro_var" or GP_name == "gp_with_oro_var_stratified":
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


def check_total_water(Q, Qs, rho):
    num = 0
    retain_reject = np.full((nlon*nlat), False)
    for i in range(len(Q[:,0])):
        water_content = np.sum(Q[i,:]*rho)
        sample = np.sum(Qs[i,:]*rho)
        diff = abs(sample - water_content)
        if diff > 1e-3:
            # Qs[i,:] = Q[i,:]
            retain_reject[i] = True
            num +=1
    print("Number of physically inconsistent profiles (total water content) %i"%num)

    return Qs, retain_reject


def check_static_energy(Q, Qs, T, Ts):
    num = 0
    retain_reject = np.full((nlon*nlat), False)
    Cp = 1.005
    Lv = 2260
    for i in range(len(Q[:,0])):
        static_energy = np.sum(Q[i,:]*Lv + Cp*T[i,:])
        sample_static_energy = np.sum(Qs[i,:]*Lv + Cp*Ts[i,:])
        diff = abs(sample_static_energy - static_energy)
        if diff > 1.0:
            # Ts[i,:] = T[i,:]
            retain_reject[i] = True
            num +=1
    print("Number of physically inconsistent profiles (moist static energy) %i"%num)

    return Ts, retain_reject


def mogp_prediction_conserving(test, trained_gp, nlon, nlat, nlev, rho):
    variance, uncer, d = trained_gp.predict(test)
    print("Prediction")
    if GP_name == "gp_without_oro_var":
        T_mean = test[:, 3:11]
        Q_mean = test[:, 11:]
    elif GP_name == "gp_with_oro_var" or GP_name == "gp_with_oro_var_stratified":
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

    resampled_Q, retain_reject_Q = check_total_water(Q_mean, resampled_Q, rho)
    resampled_T, retain_reject_T = check_static_energy(Q_mean, resampled_Q, T_mean, resampled_T)

    resampled_T = np.reshape(resampled_T, (nlon, nlat, nlev))
    resampled_T  = np.flip(resampled_T, axis = 2)
    resampled_Q = np.reshape(resampled_Q, (nlon, nlat, nlev))
    resampled_Q  = np.flip(resampled_Q, axis = 2)
    retain_reject_T = np.reshape(retain_reject_T, (nlon, nlat))
    retain_reject_Q = np.reshape(retain_reject_Q, (nlon, nlat))


    return resampled_T, resampled_Q, retain_reject_T, retain_reject_Q



trained_gp = pickle.load(open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "rb"))

# Defining constants and initial values
# SPEEDY_DATE_FORMAT = "%Y%m%d%H"
nature_dir = os.path.join(SPEEDY_root, "DATA", "nature")

IDate = "1982010100"
# dtDate = "1982010106"
# number_time_steps = (3652*4) 
nlon = 96
nlat = 48
nlev = 8
# dt = 6

# Initialisation steps
data = read_grd(os.path.join(nature_dir, IDate +".grd"), nlon, nlat, nlev)

# Read in the orography and land/sea fraction
oro = read_const_grd(os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "sfc.grd"), nlon, nlat, 0)
lsm = read_const_grd(os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "sfc.grd"), nlon, nlat, 1)
oro = np.flip(oro, 1)
lsm = np.flip(lsm, 1)
rho = np.loadtxt(os.path.join(HYBRID_root, "src", "density.txt"))
if GP_name == "gp_with_oro_var" or GP_name == "gp_with_oro_var_stratified":
    oro = np.stack((oro, read_oro_var()), axis=2)

test = data_prep(data, oro, lsm, nlon, nlat)
print("Data Prep")
print(test.shape)

resampled_T, resampled_Q, retain_reject_T, retain_reject_Q = mogp_prediction_conserving(
    test, 
    trained_gp, 
    nlon, 
    nlat, 
    nlev, 
    rho
)

pressure_levels = [30, 100, 200, 300, 500, 700, 850, 925] # hPa - top to bottom
vertical_levels = [1,2,3,4,5,6,7,8] # bottom to top

def plot_profile(ax, values_old, values_new, reject: bool, var: str):
    ax.plot(values_old, vertical_levels, color="tab:blue")
    ax.scatter(values_old, vertical_levels, color="tab:blue", label="original")
    # ax.plot(values_new, vertical_levels, color="tab:orange", linestyle="dashed")
    ax.plot(values_new, vertical_levels, color="tab:orange", linestyle=(0, (4, 3)))
    ax.scatter(values_new, vertical_levels, color="tab:orange", marker="^", label="proposal")
    ax.legend()
    if reject == False:
        title = f"{var} \n Proposal retained"
    else:
        title = f"{var} \n Proposal REJECTED"
    ax.set_title(title)


for i in range(96):
    print(f"i={i}")
    for j in range(48):
        fig, axes = plt.subplots(
            nrows=1, 
            ncols=2,
            figsize=(8, 8),
            sharey=True
        )
        fig.supylabel("Pressure level / hPa")
        plt.yticks(vertical_levels, pressure_levels[::-1])
        plot_profile(
            axes[0], 
            data[i,j,16:24], 
            resampled_T[i,j,:], 
            retain_reject_T[i,j], 
            "Temperature / K"
        )
        plot_profile(
            axes[1], 
            data[i,j,24:32], 
            resampled_Q[i,j,:], 
            retain_reject_Q[i,j], 
            "Specific Humidity / kg/kg"
        )
        fig.suptitle(f"Vertical Profiles \n Grid point: ({i+1}, {j+1})")
        plt.savefig(
            os.path.join(output_path, f'{i+1}_{j+1}.png')
        )
        plt.close()