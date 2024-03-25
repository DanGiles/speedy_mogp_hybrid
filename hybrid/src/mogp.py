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
    # index, 
    region,
    output_path: str,
    sigma = 2,
) -> None:
    pressure_levels = [30, 100, 200, 300, 500, 700, 850, 925]

    # print(region)
    region, subregion, time = region

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
    axes[0].set_ylim(bottom=0., top=1000.)
    axes[0].invert_yaxis()

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
    axes[1].set_ylim(bottom=0., top=1000.)
    axes[1].invert_yaxis()

    fig.supxlabel("Standard Deviation")
    fig.supylabel("Pressure (hPa)")
    fig.suptitle(f"Standard Deviations\nRegion: {region+1}, Subregion: {subregion+1}, Time: {(time*6):02d}:00")
    plt.legend()

    plt.savefig(
        os.path.join(output_path, f"mogp_pred_{(region+1):02d}_{subregion+1}_{(time*6):02d}.png")
    )
    plt.close()



def stratified_experimental_design(
    n_train: int
) -> np.ndarray:
    # There are 8 rows with constant latitude, these are symetric around the equator
    # The first 4 rows in the northern hemisphere have 3, 9, 13 and 15 LAMs respectively, totally 40 LAMs.
    # We need to build the stratified samples over these points. Lets do so proportionally to the number of LAMs in each row.
    lam_row_counts = [3, 9, 13, 15, 15, 13, 9, 3]
    sample_row_counts = [round(n_train * (lam_row_count/region_count)) for lam_row_count in lam_row_counts]
    running_sum = 0
    samples = np.empty((0, 3), dtype=int)
    for lam_count, sample_row_count in zip(lam_row_counts, sample_row_counts):
        ed = mogp_emulator.LatinHypercubeDesign([
            (running_sum, running_sum+lam_count), 
            (0, subregion_count), 
            (0, 4)
        ])
        
        sample = ed.sample(sample_row_count).astype(int)
        samples = np.vstack((samples, sample))
        
        running_sum += lam_count
    return samples


def hypercube(
        X: np.ndarray, 
        Y: np.ndarray, 
        oro, 
        ls, 
        n_train: int
    ):
    # Size of the training and testing datasets
    n_test = 100 # test sample size

    # # test design over region, subregion and time
    # ed = mogp_emulator.LatinHypercubeDesign([
    #     (0, region_count), 
    #     (0, subregion_count), 
    #     (0, 4)
    # ])

    # # sample space - has shape (n_train, 3)
    # train_indices = ed.sample(n_train).astype(int)

    ##### TRY THE NEW STRATIFIED DESIGN #####
    train_indices = stratified_experimental_design(n_train)

    # Training data
    site_indices = train_indices[:, 0]
    cell_indices = train_indices[:, 1]
    time_indices = train_indices[:, 2]
    X_train = X[:, :, cell_indices, site_indices, time_indices]
    Y_train = Y[:, :, cell_indices, site_indices, time_indices]
    if GP_name == "gp_without_oro_var":
        oro_train = oro[0, cell_indices, site_indices]
    elif GP_name == "gp_with_oro_var":
        oro_train = oro[:, cell_indices, site_indices]
    else:
        raise Exception("GP_name not recognised.")
    ls_train = ls[cell_indices, site_indices]

    # Testing data
    site_indices = np.random.choice(region_count, size=n_test, replace=True)
    cell_indices = np.random.choice(subregion_count, size=n_test, replace=True)
    time_indices = np.random.choice(4, size=n_test, replace=True)
    X_test = X[:, :, cell_indices, site_indices, time_indices]
    Y_test = Y[:, :, cell_indices, site_indices, time_indices]
    if GP_name == "gp_without_oro_var":
        oro_test = oro[0, cell_indices, site_indices]
    elif GP_name == "gp_with_oro_var":
        oro_test = oro[:, cell_indices, site_indices]
    else:
        raise Exception("GP_name not recognised.")
    ls_test = ls[cell_indices, site_indices]

    test_indices = np.vstack((site_indices, cell_indices, time_indices)).T
    # print(test_indices.shape, test_indices.dtype)

    return X_train, X_test, Y_train, Y_test, oro_train, oro_test, ls_train, ls_test, train_indices, test_indices


def crop_speedy(array: np.ndarray) -> np.ndarray:
    "Pressure Levels in Speedy defined on pressure levels"
    "30, 100, 200, 300, 500, 700, 850 and 925 hPa"

    pressure_level = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]
    indices = np.zeros(len(pressure_level), dtype=int)
    for j in range(len(pressure_level)):
        indices[j] = np.argmin(np.abs(pressure_level[j] - array[0, :, 0]))
    return indices


def data_prep(X, X_ps, oro, ls, y) -> Tuple[np.ndarray, np.ndarray]:
    if GP_name == "gp_without_oro_var":
        train = np.empty((19, X.shape[2]), dtype = np.float64)

        train[0, :] = X_ps  #surface level AVG air pressure
        train[1, :] = oro   #orography
        train[2, :] = ls    #land-sea ratio
        train[3:11, :] = X[0, :, :] #AVG air temp at desired levels
        train[11:, :] = X[1, :, :]  #AVG humudity at desired levels
    elif GP_name == "gp_with_oro_var":
        train = np.empty((20, X.shape[2]), dtype = np.float64)

        train[0, :] = X_ps  #surface level AVG air pressure
        train[1:3, :] = oro   #orography
        train[3, :] = ls    #land-sea ratio
        train[4:12, :] = X[0, :, :] #AVG air temp at desired levels
        train[12:, :] = X[1, :, :]  #AVG humudity at desired levels
    else:
        raise Exception("GP_name not recognised.")
    
    target = np.empty((16, X.shape[2]), dtype = np.float64)
    target[:8, :] = y[0, :] #STD air temp at desired levels
    target[8:, :] = y[1, :] #STD humudity at desired levels

    return train, target


def train_mogp(n_train):
    X = np.load(os.path.join(processed_data_root, "20200101_mean.npy"))
    Y = np.load(os.path.join(processed_data_root, "20200101_std.npy"))
    # print(X.shape, Y.shape)

    print("Loaded in the X and Y")
    oro = np.load(os.path.join(processed_data_root, "20200101_orography.npy"))
    land_sea = np.load(os.path.join(processed_data_root, "20200101_land_sea.npy"))

    #X_train.shape:     (3, UM_levels, n_train)
    #X_test.shape:      (3, UM_levels, n_test)
    #y_train.shape:     (2, UM_levels, n_train)
    #y_test.shape:      (2, UM_levels, n_test)
    #oro_train.shape:   (n_train, ) or (2, n_train)
    #oro_test.shape:    (n_test, ) or (2, n_test)
    #ls_train.shape:    (n_train, )
    #ls_test.shape:     (n_test, )
    X_train, X_test, y_train, y_test, oro_train, oro_test, ls_train, ls_test, train_indices, test_indices = hypercube(X, Y, oro, land_sea, n_train)

    #extract mean air pressure at surface level at all locations
    #X_train_ps.shape:  (n_train, )
    #X_test_ps.shape:   (n_test, )
    X_train_ps = X_train[0, 0, :]
    X_test_ps = X_test[0, 0, :]

    #extract air temp (and humidity) at desired levels at all locations
    print("Cropping arrays")
    indices = crop_speedy(X_train)

    #X_train.shape:     (2, 8, n_train)
    #X_test.shape:      (2, 8, n_test)
    #y_train.shape:     (2, 8, n_train)
    #y_test.shape:      (2, 8, n_test)
    X_train = X_train[1:, indices, :]
    X_test = X_test[1:, indices, :]
    y_train = y_train[:, indices, :]
    y_test = y_test[:, indices, :]

    # # Setting up the training dataset
    input, target = data_prep(X_train, X_train_ps, oro_train, ls_train, y_train)
    # input = np.load(os.path.join(output_folder, "input.npy"))
    # target = np.load(os.path.join(output_folder, "target.npy"))
    print("input and target", input.shape, target.shape)
    # np.save(os.path.join(output_folder, "input.npy"), input)
    # np.save(os.path.join(output_folder, "target.npy"), target)


    # This switch is primarily used for my testing
    if TRAIN_GP is True:
        # # Defining and fitting the MOGP
        #gp = mogp_emulator.MultiOutputGP(input.T, target, kernel="SquaredExponential")
        gp = mogp_emulator.MultiOutputGP(input.T, target, kernel="Matern52")
        gp = mogp_emulator.fit_GP_MAP(gp)
        # # Save the trained mogp
        np.save(
            os.path.join(gp_directory_root, f"{GP_name}_train_indices.npy"), 
            train_indices
        )
        pickle.dump(gp, open(os.path.join(gp_directory_root, f"{GP_name}.pkl"),"wb"))
    else:
        #Read in the pre-trained GP
        gp = pickle.load(open(os.path.join(gp_directory_root, f"{GP_name}.pkl"), "rb"))






    # # Setting up the testing dataset
    test, truth = data_prep(X_test, X_test_ps, oro_test, ls_test, y_test)
    print("test and truth", test.shape, truth.shape)
    # Loading the trained mogp from file. Not needed but used to test implementation
    # test = np.load(os.path.join(output_folder, "test.npy"))
    # truth = np.load(os.path.join(output_folder, "truth.npy"))
    # np.save(os.path.join(output_folder, "test.npy"), test)
    # np.save(os.path.join(output_folder, "truth.npy"), truth)

    # Predict using the MOGP
    variances, uncer, d = gp.predict(test.T)
    # print(test.dtype, variances.dtype, truth.dtype)

    output_path = os.path.join(pngs_root, GP_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Save the test indices
    np.save(
        os.path.join(output_path, "test_indices.npy"), 
        test_indices
    )
    for test_index in range(test_indices.shape[0]):
        plot_mogp_predictions(
            truth[:8, test_index],
            truth[8:, test_index],
            variances[:8, test_index], uncer[:8, test_index],
            variances[8:, test_index], uncer[8:, test_index],
            # test_index,
            test_indices[test_index, :],
            output_path
        )


if __name__ == '__main__':
    train_mogp(500)
