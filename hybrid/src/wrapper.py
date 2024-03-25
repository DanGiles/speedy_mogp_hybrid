#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import subprocess
from datetime import datetime, date, timedelta
import shutil # for removing data from previous simulations
import pickle

from mogp import *
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

def speedy_update(SPEEDY, output_folder, YMDH, TYMDH):
    # Path to the bash script which carries out the forecast
    forecast = os.path.join(HYBRID_root, "src", "dafcst.sh")
    # Bash script call to speedy
    subprocess.check_call(str(forecast)+" %s %s %s %s" % (str(SPEEDY), str(output_folder), str(YMDH), str(TYMDH)),shell=True)
    return

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


def mogp_prediction(mogp_inputs, trained_gp, nlon, nlat, nlev):
    variance, uncer, d = trained_gp.predict(mogp_inputs)
    print("Prediction")
    if GP_name == "gp_without_oro_var":
        T_mean = mogp_inputs[:, 3:11]
        Q_mean = mogp_inputs[:, 11:]
    elif GP_name == "gp_with_oro_var":
        T_mean = mogp_inputs[:, 4:12]
        Q_mean = mogp_inputs[:, 12:]
    resampled_T = np.empty((nlon*nlat*nlev), dtype = np.float64)
    resampled_Q = np.empty((nlon*nlat*nlev), dtype = np.float64)
    
    low_values_flags = variance < 1e-6  # Where values are low
    variance[low_values_flags] = 0.0

    draws = np.random.normal(0, 1, np.shape(T_mean.flatten()))
    resampled_T = T_mean.flatten() + draws * (variance[:8,:].T.flatten() + uncer[:8,:].T.flatten())
    resampled_Q = Q_mean.flatten() + draws * (variance[8:,:].T.flatten() + uncer[8:,:].T.flatten())

    resampled_Q = np.reshape(resampled_Q.T, (nlon*nlat, nlev))
    resampled_T = np.reshape(resampled_T.T, (nlon*nlat, nlev))

    resampled_T = np.reshape(resampled_T, (nlon, nlat, nlev))
    resampled_Q = np.reshape(resampled_Q, (nlon, nlat, nlev))

    return resampled_T, resampled_Q



def main():
    if TRAIN_GP:
        # Train the GP Model
        plot_folder = os.path.join("", "")
        n_train = 500
        print("Starting Training")
        trained_gp, test_UM = train_mogp(plot_folder, n_train)
    else:
        # Read in pre-trained GP model
        trained_gp = pickle.load(open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "rb"))
    print(trained_gp)
    print("Training Done!")

    # Defining constants and initial values
    SPEEDY_DATE_FORMAT = "%Y%m%d%H"
    nature_dir = os.path.join(SPEEDY_root, "DATA", "nature")

    IDate = "1982010100"
    dtDate = "1982010106"
    number_time_steps = (3652*4) 
    nlon = 96
    nlat = 48
    nlev = 8
    dt = 6
    
    # Initialisation steps
    data_folder = os.path.join(HYBRID_data_root, GP_name)
    create_folders(data_folder)
    data = read_grd(os.path.join(nature_dir, IDate +".grd"), nlon, nlat, nlev)

    # Read in the orography and land/sea fraction
    oro = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 0)
    lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
    oro = np.flip(oro, 1)
    lsm = np.flip(lsm, 1)

    if GP_name == "gp_with_oro_var":
        oro = np.stack((oro, read_oro_var()), axis=2)

    # Main time loop
    for t in range(0,number_time_steps):
        # Time counters
        print(IDate, dtDate, t)

        # Time to do the MOGP magic
        # Loop through all columns
        mogp_inputs = data_prep(data, oro, lsm, nlon, nlat)
        print("Data Prep")
        resampled_T, resampled_Q = mogp_prediction(mogp_inputs, trained_gp, nlon, nlat, nlev)
        print("Max T Difference %f"%(np.amax(data[:,:,16:24] - resampled_T[:,:,:])))
        print("Max Q Difference %f"%(np.amax(data[:,:,24:32] - resampled_Q[:,:,:])))
        data[:,:,16:24] = resampled_T[:,:,:]
        data[:,:,24:32] = resampled_Q[:,:,:]

        # # Write updated data to fortran speedy file
        file = os.path.join(data_folder, (IDate+".grd"))
        print("Writing file")
        write_fortran(file, data)
        print("Done Writing")

        # # # Speedy integration forward
        speedy_update(SPEEDY_root, data_folder, IDate, dtDate)

        # # # Read Speedy output
        file = os.path.join(data_folder, (dtDate+".grd"))
        data = read_grd(file, nlon, nlat, nlev)
        # # Update time counters
        IDate, dtDate = step_datetime(IDate, dtDate, SPEEDY_DATE_FORMAT, dt)

    return

if __name__ == '__main__':
    main()


