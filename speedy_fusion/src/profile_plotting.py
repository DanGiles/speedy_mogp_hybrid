#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt



def profile(mean_T, variance_t, mean_est, variance_est, figname):
    pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]

    fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
    # axs[0].plot(mean_T, pressure_levels, color='k', label="$\mu")
    axs[0].plot((variance_t), pressure_levels, 'o', color='red', label="True $+\sigma$")
    axs[0].plot((-variance_t), pressure_levels, 'o', color='red', label="True $-\sigma$")
    axs[0].set_ylabel("Pressure (Pa)")
    axs[0].set_title("True Variance at each level")


    # axs[1].plot(mean_est, pressure_levels, color='k', label="$\mu")
    axs[1].plot((variance_est), pressure_levels, 'x', color='blue', label="Predicted $+\sigma$")
    axs[1].plot((-variance_est), pressure_levels, 'x', color='blue', label="Predicted $-\sigma$")
    plt.savefig(figname, format='png', bbox_inches='tight')
    axs[1].set_title("Predicted Variance at each level")
    plt.show(block=False)

    return


def single_profile(variance_t, variance_est, figname):
    pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]

    fig, axs = plt.subplots(1, 1, sharex = True, sharey = True)
    # axs[0].plot(mean_T, pressure_levels, color='k', label="$\mu")
    axs.plot((variance_t), pressure_levels, 'o', color='red', label="True $+\sigma$")
    axs.plot((-variance_t), pressure_levels, 'o', color='red', label="True $-\sigma$")
    axs.plot((variance_est), pressure_levels, 'x', color='blue', label="Predicted $+\sigma$")
    axs.plot((-variance_est), pressure_levels, 'x', color='blue', label="Predicted $-\sigma$")
    axs.set_ylabel("Pressure (Pa)")
    axs.set_xlabel("Variance")
    axs.set_title("Variance at each level")
    plt.savefig(figname, format='png', bbox_inches='tight')

    plt.show(block=False)
    plt.close()
    return


