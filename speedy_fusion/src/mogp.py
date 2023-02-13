#!/bin/zsh
# -*- coding: utf-8 -*-
import os
import numpy as np 
import mogp_emulator
import dill
# Load internal functions
from profile_plotting import *
# from loop_files import *
# from data_prep import *
# from dgp import *
# from prep_UM import *

def hypercube(X, Y, oro, ls, train):
    # Size of the training and testing datasets
    test = 100
    ed = mogp_emulator.LatinHypercubeDesign([(0, 94), (0, 196), (0, 4)])
    # sample space
    inputs = ed.sample(train).astype(int)
    site_indices = inputs[:,0]
    cell_indices = inputs[:,1]
    time_indices = inputs[:,2]
    X_train = X[:,:,cell_indices, site_indices, time_indices]
    Y_train = Y[:,:,cell_indices, site_indices, time_indices]
    oro_train = oro[0, cell_indices, site_indices]
    ls_train = ls[0, cell_indices, site_indices]

    # Setting up the training dataset
    site_indices = np.random.choice(94, size=test, replace=True)
    cell_indices = np.random.choice(196, size=test, replace=True)
    time_indices = np.random.choice(4, size=test, replace=True)
    X_test = X[:,:,cell_indices, site_indices, time_indices]
    Y_test = Y[:,:,cell_indices, site_indices, time_indices]
    oro_test = oro[0, cell_indices, site_indices]
    ls_test = ls[0, cell_indices, site_indices]

    return X_train, X_test, Y_train, Y_test, oro_train, oro_test, ls_train, ls_test


def crop_speedy(array):
    "Pressure Levels in Speedy defined on pressure levels"
    "30, 100, 200, 300, 500, 700, 850 and 925 hPa"

    pressure_level = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]
    # pressure_level *= 100
    # for ex in range(len(array[0,0,:])):
    indices = np.zeros(len(pressure_level), dtype=int)
    for j in range(len(pressure_level)):
        indices[j] = np.argmin(np.abs(pressure_level[j]-array[0,:,0]))
    return indices

def data_prep(X, X_ps, oro, ls, y):
    train = np.empty((19, X.shape[2]), dtype = np.float64)
    target = np.empty((16, X.shape[2]), dtype = np.float64)
    train[0,:] = X_ps
    train[1,:] = oro
    train[2,:] = ls
    train[3:11,:] = X[0,:]
    train[11:,:] = X[1,:]
    target[:8,:] = y[0,:]
    target[8:,:] = y[1,:]
    return train, target


def train_mogp(output_folder, n_train):
    # output_folder = "../speedy_layers/temp_profile"    
    #X = np.load('/home/ucakdpg/Scratch/mogp-speedy/processed/r1_100_t4/mean.npy')
    #Y = np.load('/home/ucakdpg/Scratch/mogp-speedy/processed/r1_100_t4/std.npy')
    #print(X.shape, Y.shape)

    print("Loaded in the X and Y")
    #oro = np.load('/home/ucakdpg/Scratch/mogp-speedy/processed/r1_100_t4/orography.npy')
    #land_sea = np.load('/home/ucakdpg/Scratch/mogp-speedy/processed/r1_100_t4/land_sea.npy')

    # X_train, X_test, y_train, y_test, oro_train, oro_test, ls_train, ls_test = splitting_spatial_temporal(X, Y, oro, land_sea)
    #X_train, X_test, y_train, y_test, oro_train, oro_test, ls_train, ls_test = hypercube(X, Y, oro, land_sea, n_train)
    #print(X_train.shape)
    #print("Cropping arrays")
    #indices = crop_speedy(X_train)
    #X_train_ps = X_train[0,0,:]
    #X_test_ps = X_test[0,0,:]

    #X_train = X_train[1:,indices,:]
    #X_test = X_test[1:,indices,:]
    #y_train = y_train[1:,indices,:]
    #y_test = y_test[1:,indices,:]

    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, oro_test.shape)  

    input_layers = 9
    variables = ['T', 'Q']
    # # Setting up the training dataset
    #input, target = data_prep(X_train, X_train_ps, oro_train, ls_train, y_train)
    input = np.load(os.path.join(output_folder, "input.npy"))
    target = np.load(os.path.join(output_folder, "target.npy"))
    print("input and target", input.shape, target.shape)
    #np.save(os.path.join(output_folder, "input.npy"), input)
    #np.save(os.path.join(output_folder, "target.npy"), target)
    # # Defining and fitting the MOGP
    #gp = mogp_emulator.MultiOutputGP(input.T, target, kernel="SquaredExponential")
    gp = mogp_emulator.MultiOutputGP(input.T, target, kernel="Matern52")
    gp = mogp_emulator.fit_GP_MAP(gp)
    # # Save the trained mogp
    dill.dump(gp, open(os.path.join(output_folder, "gp.pkl"),"wb"))
    # # Setting up the testing dataset
    #test, truth = data_prep(X_test, X_test_ps, oro_test, ls_test, y_test)
    #print("test and truth", test.shape, truth.shape)
    # Loading the trained mogp from file. Not needed but used to test implementation
    test = np.load(os.path.join(output_folder, "test.npy"))
    truth = np.load(os.path.join(output_folder, "truth.npy"))
    #np.save(os.path.join(output_folder, "test.npy"), test)
    #np.save(os.path.join(output_folder, "truth.npy"), truth)
    # Predict using the MOGP
    variances, uncer, d = gp.predict(test.T)
    print(test.dtype, variances.dtype, truth.dtype)
    count = 0
    for var in variables:
        for i in range(99):
            figname = os.path.join(output_folder, "mogp_"+var+"_%i.png"%i)
            if count == 0 :
                # print(truth[:8,i], variances[:8,i], uncer[:8,i])
                single_profile(truth[:8,i], variances[:8,i], figname)
            else:
                # print(truth[8:,i], variances[8:,i], uncer[8:,i])
                single_profile(truth[8:,i], variances[8:,i], figname)
        count = count +1

    return gp, test

if __name__ == '__main__':
    main()
