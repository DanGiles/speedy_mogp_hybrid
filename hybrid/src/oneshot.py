#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil # for removing data from previous simulations
import pickle

from mogp import *
from script_variables import *
    

def create_folders(output_folder):
    tmp = os.path.join(output_folder, "tmp")

    #do not empty directory if it doesn't exist!
    if os.path.isdir(tmp):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    os.mkdir(tmp)


def read_grd(filename, nlon, nlat, nlev) -> np.ndarray:
    """Reads in a SPEEDY .grd file and returns a numpy array of the data.
    Use this function for reading in SPEEDY output files.

    Args:
        filename (str): The path to the .grd file
        nlon (int): The number of longitudes
        nlat (int): The number of latitudes
        nlev (int): The number of levels
    """

    nv3d = 4
    nv2d = 2
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,nv3d*nlev+nv2d)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data

def read_const_grd(filename, nlon, nlat, var):
    """Reads in a SPEEDY .grd file and returns a numpy array of the data.
    Use this function for reading in the orography and land/sea mask.

    Args:
        filename (str): The path to the .grd file
        nlon (int): The number of longitudes
        nlat (int): The number of latitudes
        var (int): The variable to extract
    """
    # argument var accepts the following values:
    #   Orography = 0
    #   Land/Sea Mask = 1

    num = 5
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data[:,:,var]


def read_oro_var() -> np.ndarray:
    """Reads in the orography variance data for SPEEDY."""
    oro_var_data = np.zeros((96, 48))
    oro_var_data_file = os.path.join(SPEEDY_root, "model","data","bc","t30","clim", "std_orog_for_speedy.dat")
    with open(oro_var_data_file) as f:
        for row_i in range(96):
            oro_var_data[row_i, :] = np.fromstring(f.readline().strip(), dtype=float, sep=',')
    return oro_var_data


def data_prep(data, oro, ls, nlon, nlat) -> np.ndarray:
    """Prepares the data for prediction with the MOGP model.
    
    Args:
        data (np.ndarray): The SPEEDY output data
        oro (np.ndarray): The orography data
        ls (np.ndarray): The land/sea mask data
        nlon (int): The number of longitudes
        nlat (int): The number of latitudes
    """
    T_mean = data[:,:,16:24] # Temperature for all levels
    Q_mean = data[:,:,24:29] # Specific Humidity for bottom 5 levels

    low_values_flags = Q_mean[:,:] < 1e-6  # Where values are low
    Q_mean[low_values_flags] = 1e-6

    # Version for gp_with_oro_var
    # first axis is number of spatial grid points
    # second axis is number of features
    # surface level pressure, orography (mean and var), land/sea mask, temperature x8, specific humidity x5
    train = np.empty(((nlon*nlat),17), dtype = np.float64)
    train[:, 0] = data[:,:,32].flatten()
    train[:, 1] = oro[...,0].flatten()
    train[:, 2] = oro[...,1].flatten() 
    train[:, 3] = ls.flatten()
    train[:, 4:12] = np.reshape(T_mean, ((nlon*nlat), 8))
    train[:, 12:] = np.reshape(Q_mean, ((nlon*nlat), 5))

    return train


def mogp_prediction(mogp_inputs, trained_gp, nlon, nlat, nlev):
    """Makes a prediction with the MOGP model.
    
    Args:
        mogp_inputs (np.ndarray): The input data for making a prediction
        trained_gp (mogp_emulator model): The trained MOGP model
        nlon (int): The number of longitudes
        nlat (int): The number of latitudes
        nlev (int): The number of levels
    """
    variance, uncer, d = trained_gp.predict(mogp_inputs)
    print("Prediction")

    # Extract current atmopsheric state
    T_mean = mogp_inputs[:, 4:12]
    Q_mean = mogp_inputs[:, 12:]
    # Setup empty arrays for new atmospheric state
    resampled_T = np.empty((nlon*nlat*nlev), dtype = np.float64)
    resampled_Q = np.empty((nlon*nlat*nlev), dtype = np.float64)
    
    # # Rescale the Specific Humidity
    variance[8:,:] = variance[8:,:]/1000
    uncer[8:,:] = uncer[8:,:]/1000

    low_values_flags = variance < 1e-6  # Where values are low
    variance[low_values_flags] = 0.0

    ##### Generate the new atmospheric state
    # Generate random Gaussian perturbations of shape (nlon*nlat, nlev)
    draws = np.random.normal(0, 1, np.shape(T_mean))
    # New atmopsheric state is the current state plus the pertubration times (the variance plus the uncertainty in the variance)
    resampled_T = T_mean.flatten() + draws.flatten() * (variance[:8,:].T.flatten() + uncer[:8,:].T.flatten())
    resampled_Q = Q_mean.flatten() + draws[:,:5].flatten() * (variance[8:,:].T.flatten() + uncer[8:,:].T.flatten())

    resampled_Q = np.reshape(resampled_Q.T, (nlon*nlat, 5))
    resampled_T = np.reshape(resampled_T.T, (nlon*nlat, nlev))

    resampled_Q = np.reshape(resampled_Q, (nlon, nlat, 5))
    resampled_T = np.reshape(resampled_T, (nlon, nlat, nlev))
    # print(f"Max std T = {np.amax(variance[:8,:])} and Q = {np.amax(variance[8:,:])}")

    return variance, uncer, resampled_T, resampled_Q


def main():
    HYBRID_data_root = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/robustness_runs"
    SPEEDY_root = "/Users/dangiles/Documents/Stats/MetOffice/hybrid_modelling/speedy_mogp_hybrid/speedy"

    trained_gp = pickle.load(open(os.path.join(HYBRID_data_root, f"{GP_name}.pkl"), "rb"))
    print(trained_gp)
    print("Loading traing GP complete!")

    # Defining constants and initial values
    SPEEDY_DATE_FORMAT = "%Y%m%d%H"
    nature_dir = os.path.join(SPEEDY_root, "DATA", "nature")
    oneshot_dir = os.path.join(HYBRID_data_root, "oneshot")

    IDate = "1987010100"
    dtDate = "1987010106"
    nlon = 96
    nlat = 48
    nlev = 8
    dt = 6
    
    data = read_grd(os.path.join(nature_dir, IDate +".grd"), nlon, nlat, nlev)

    # Read in the orography and land/sea fraction
    oro = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 0)
    lsm = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sfc.grd"), nlon, nlat, 1)
    oro = np.flip(oro, 1)
    lsm = np.flip(lsm, 1)
    oro = np.stack((oro, read_oro_var()), axis=2)

    # Time counters
    print(IDate, dtDate)

    # Time to do the MOGP magic
    mogp_input = data_prep(data, oro, lsm, nlon, nlat)
    print("Data Prep")
    variance, uncer, resampled_T, resampled_Q = mogp_prediction(mogp_input, trained_gp, nlon, nlat, nlev)
    print("Max T Difference %f"%(np.amax(data[:,:,16:24] - resampled_T[:,:,:])))
    print("Max Q Difference %f"%(np.amax(data[:,:,24:29] - resampled_Q[:,:,:])))

    # Write the new data to a file
    np.save(os.path.join(oneshot_dir, f"{IDate}_variance.npy"), variance)
    np.save(os.path.join(oneshot_dir, f"{IDate}_uncert.npy"), uncer)

    return

if __name__ == '__main__':
    main()
