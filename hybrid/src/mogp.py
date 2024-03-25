#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import os
import numpy as np 
import mogp_emulator
import pickle

import matplotlib.pyplot as plt

from script_variables import *


def plot_mogp_predictions(
    UM_t_var, 
    UM_q_var,
    MOGP_t_var, MOGP_t_unc,
    MOGP_q_var, MOGP_q_unc,
    region, 
    indices,
    output_path: str,
    sigma = 2,
) -> None:
    pressure_levels = [925, 850, 700, 500, 300, 200, 100, 30]

    time, day = indices
    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 8),
        sharey=True,
    )

    # print("Region: ", region)
    # print(MOGP_t_var, MOGP_t_unc)
    # print(MOGP_q_var, MOGP_q_unc)

    axes[0].set_title('Temperature / K')
    axes[0].errorbar(
        (MOGP_t_var), 
        pressure_levels,
        xerr=np.maximum(0, MOGP_t_var+sigma*MOGP_t_unc), 
        fmt='o',
        label="MOGP"
    )
    axes[0].plot(
        (UM_t_var), 
        pressure_levels,
        'x', 
        color='red', 
        label="UM 'Truth'"
    )
    axes[0].set_xlim(left=0.)
    axes[0].set_ylim(bottom=1000., top=0.)

    axes[1].set_title('Specific Humidity / kg/kg')
    axes[1].errorbar(
        (MOGP_q_var), 
        pressure_levels,
        xerr=np.maximum(0, MOGP_q_var+sigma*MOGP_q_unc), 
        fmt='o',
        label="MOGP"
    )
    axes[1].plot(
        (UM_q_var), 
        pressure_levels,
        'x', 
        color='red', 
        label="UM 'Truth'"
    )
    axes[1].set_xlim(left=0.)
    axes[1].set_ylim(bottom=1000., top=0.)

    fig.supxlabel("Standard Deviation")
    fig.supylabel("Pressure (hPa)")
    fig.suptitle(f"Standard Deviations\nRegion: {region+1}, Day: {(day+1):02d}, Time: {(time*6):02d}:00")
    plt.legend()

    plt.savefig(
        os.path.join(output_path, f"mogp_pred_{(region+1):02d}_{(day+1):02d}_{(time*6):02d}.png")
    )
    plt.close()

def sampler(
        X:np.ndarray, 
        Y:np.ndarray,
        oro:np.ndarray,
        ls:np.ndarray,
        n_size: int
    ):

    rng = np.random.default_rng()
    regions = region_count*subregion_count
    X_split = np.zeros((3, UM_levels, regions, n_size))
    Y_split = np.zeros((2, UM_levels, regions, n_size))
    oro_split = np.zeros((2, regions, n_size))
    ls_split = np.zeros((regions, n_size))

    indx = np.zeros((regions, n_size, 2))

    for region in range(regions):
        indices = np.unravel_index(rng.integers((num_timesteps*num_days), size=n_size), (num_timesteps, num_days))
        X_split[:, :, region, :] = X[:, :, region, indices[0], indices[1]]
        Y_split[:, :, region, :] = Y[:, :, region, indices[0], indices[1]]
        oro_split[:, region, :] =  np.repeat(oro[:, region, np.newaxis], n_size, axis=1)
        ls_split[region, :] = ls[region]
        indx[region, :, 0] = indices[0]
        indx[region, :, 1] = indices[1]

    X_split = np.reshape(X_split, (3, UM_levels, regions*n_size))
    Y_split = np.reshape(Y_split, (2, UM_levels, regions*n_size))
    oro_split = np.reshape(oro_split, (2, regions*n_size))
    ls_split = np.reshape(ls_split, (regions*n_size))

    return X_split, Y_split, oro_split, ls_split, indx.astype(int)

def crop_speedy(array: np.ndarray, sigma_levels: np.ndarray) -> np.ndarray:
    indices = np.zeros(len(sigma_levels), dtype=int)
    # Convert to sigma coordinates
    array = array/array[0]

    for j in range(len(sigma_levels)):
        indices[j] = np.argmin(np.abs(sigma_levels[j] - array))
    return indices

def map_to_speedy_pressure_levels(X: np.ndarray, Y: np.ndarray):
    "Pressure Levels in Speedy defined on sigma levels"
    # pressure_level = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]
    sigma_levels = [0.95, 0.835, 0.685, 0.51, 0.34, 0.20, 0.095, 0.025]
    # sigma_levels = [0.025, 0.095, 0.2, 0.34, 0.51, 0.685, 0.835, 0.95]
    # pressure_level = np.flip(pressure_level)
    X_new = np.zeros((3, len(sigma_levels), X.shape[2]))
    Y_new = np.zeros((2, len(sigma_levels), Y.shape[2]))

    for region in range(X.shape[2]):
        indices = crop_speedy(X[0, :, region], sigma_levels)
        X_new[:,:,region] = X[:,indices,region]
        Y_new[:,:,region] = Y[:,indices,region]

    return X_new, Y_new


def data_prep(X, X_ps, oro, ls, y) -> Tuple[np.ndarray, np.ndarray]:
    if GP_name == "gp_without_oro_var":
        train = np.empty((19, X.shape[2]), dtype = np.float64)

        train[0, :] = X_ps  #surface level AVG air pressure
        train[1, :] = oro   #orography
        train[2, :] = ls    #land-sea ratio
        train[3:11, :] = X[1, :, :] #AVG air temp at desired levels
        train[11:, :] = X[2, :, :]  #AVG humudity at desired levels
    elif GP_name == "gp_with_oro_var":
        train = np.empty((20, X.shape[2]), dtype = np.float64)

        train[0, :] = X_ps  #surface level AVG air pressure
        train[1:3, :] = oro   #orography
        train[3, :] = ls    #land-sea ratio
        train[4:12, :] = X[1, :, :] #AVG air temp at desired levels
        train[12:, :] = X[2, :, :]  #AVG humudity at desired levels
    else:
        raise Exception("GP_name not recognised.")
    
    target = np.empty((16, X.shape[2]), dtype = np.float64)
    target[:8, :] = y[0, :] #STD air temp at desired levels
    target[8:, :] = y[1, :] #STD humudity at desired levels

    return train, target

def loop_through_days(processed_data_root):
    X = np.zeros(
        (3, UM_levels, (region_count*subregion_count), num_timesteps, num_days)
    )
    #(3, UM_levels, subregion_count, region_count, len(time))
    Y = np.zeros(
        (2, UM_levels, (region_count*subregion_count), num_timesteps, num_days)
    )
    for day in range(0,num_days):
        X[...,day] = np.load(os.path.join(processed_data_root, f"202001{(day+1):02d}_mean.npy"))
        Y[...,day] = np.load(os.path.join(processed_data_root, f"202001{(day+1):02d}_std.npy"))

    return X, Y

def train_mogp():

    X, Y = loop_through_days(processed_data_root)

    oro = np.load(os.path.join(processed_data_root, "20200101_orography.npy"))
    land_sea = np.load(os.path.join(processed_data_root, "20200101_land_sea.npy"))

    # #X_train.shape:     (3, UM_levels, n_train)
    # #X_test.shape:      (3, UM_levels, n_test)
    # #y_train.shape:     (2, UM_levels, n_train)
    # #y_test.shape:      (2, UM_levels, n_test)
    # #oro_train.shape:   (n_train, ) or (2, n_train)
    # #oro_test.shape:    (n_test, ) or (2, n_test)
    # #ls_train.shape:    (n_train, )
    # #ls_test.shape:     (n_test, )
    X_train, Y_train, oro_train, ls_train, train_indices= sampler(X, Y, oro, land_sea, n_size=1)
    X_test, Y_test, oro_test, ls_test, test_indices = sampler(X, Y, oro, land_sea, n_size=1)

    # #extract air temp (and humidity) at desired levels at all locations
    # print("Cropping arrays")
    X_train, Y_train = map_to_speedy_pressure_levels(X_train, Y_train)
    X_test, Y_test = map_to_speedy_pressure_levels(X_test, Y_test)
    print(X_train.shape, Y_train.shape, oro_train.shape, ls_train.shape)
    #extract mean air pressure at surface level at all locations
    # #X_train_ps.shape:  (n_train, )
    # #X_test_ps.shape:   (n_test, )
    X_train_ps = X_train[0, 0, :]
    X_test_ps = X_test[0, 0, :]

    # # # Setting up the training dataset
    train_input, train_target = data_prep(X_train, X_train_ps, oro_train, ls_train, Y_train)
    # # input = np.load(os.path.join(output_folder, "input.npy"))
    # # target = np.load(os.path.join(output_folder, "target.npy"))
    print("input and target", train_input.shape, train_target.shape)
    # # np.save(os.path.join(output_folder, "input.npy"), input)
    # # np.save(os.path.join(output_folder, "target.npy"), target)

    # This switch is primarily used for my testing
    if TRAIN_GP is True:
        # # Defining and fitting the MOGP
        gp = mogp_emulator.MultiOutputGP(train_input.T, train_target, kernel="Matern52", nugget='adaptive')
        gp = mogp_emulator.fit_GP_MAP(gp)
        # # Save the trained mogp
        pickle.dump(gp, open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "wb"))
    else:
        #Read in the pre-trained GP
        print(gp_directory_root)
        gp = pickle.load(open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "rb"))


    # # Setting up the testing dataset
    test_input, test_target = data_prep(X_test, X_test_ps, oro_test, ls_test, Y_test)
    print("test and truth", test_input.shape, test_target.shape)
    # Loading the trained mogp from file. Not needed but used to test implementation
    np.save(os.path.join(gp_directory_root, "test_input.npy"), test_input)
    np.save(os.path.join(gp_directory_root, "test_target.npy"), test_target)
    # test_target = np.load(os.path.join(gp_directory_root, "test_target.npy"))
    # Predict using the MOGP
    stds, uncer, d = gp.predict(test_input.T)
    np.save(os.path.join(gp_directory_root, "stds.npy"), stds)
    np.save(os.path.join(gp_directory_root, "uncer.npy"), uncer)

    # stds = np.load(os.path.join(gp_directory_root, "stds.npy"))
    # uncer = np.load(os.path.join(gp_directory_root, "uncer.npy"))
    output_path = os.path.join(pngs_root, GP_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    print(test_target.shape)


    for test_index in range(test_target.shape[1]):
        plot_mogp_predictions(
            test_target[:8, test_index],
            test_target[8:, test_index],
            stds[:8, test_index], uncer[:8, test_index],
            stds[8:, test_index], uncer[8:, test_index],
            test_index,
            test_indices[test_index, 0, :],
            output_path
        )


if __name__ == '__main__':
    train_mogp()
