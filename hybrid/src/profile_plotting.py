#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt



# def profile(mean_T, variance_t, mean_est, variance_est, figname):
#     pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]

#     fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
#     # axs[0].plot(mean_T, pressure_levels, color='k', label="$\mu")
#     axs[0].plot((variance_t), pressure_levels, 'o', color='red', label="True $+\sigma$")
#     axs[0].plot((-variance_t), pressure_levels, 'o', color='red', label="True $-\sigma$")
#     axs[0].set_ylabel("Pressure (Pa)")
#     axs[0].set_title("True Variance at each level")


#     # axs[1].plot(mean_est, pressure_levels, color='k', label="$\mu")
#     axs[1].plot((variance_est), pressure_levels, 'x', color='blue', label="Predicted $+\sigma$")
#     axs[1].plot((-variance_est), pressure_levels, 'x', color='blue', label="Predicted $-\sigma$")
#     plt.savefig(figname, format='png', bbox_inches='tight')
#     axs[1].set_title("Predicted Variance at each level")
#     plt.show(block=False)

#     return


# def single_profile(variance_t, variance_est, figname):
#     pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]

#     fig, axs = plt.subplots(1, 1, sharex = True, sharey = True)
#     # axs[0].plot(mean_T, pressure_levels, color='k', label="$\mu")
#     axs.plot((variance_t), pressure_levels, 'o', color='red', label="True $+\sigma$")
#     axs.plot((-variance_t), pressure_levels, 'o', color='red', label="True $-\sigma$")
#     axs.plot((variance_est), pressure_levels, 'x', color='blue', label="Predicted $+\sigma$")
#     axs.plot((-variance_est), pressure_levels, 'x', color='blue', label="Predicted $-\sigma$")
#     axs.set_ylabel("Pressure (Pa)")
#     axs.set_xlabel("Variance")
#     axs.set_title("Variance at each level")
#     plt.savefig(figname, format='png', bbox_inches='tight')

#     plt.show(block=False)
#     plt.close()
#     return


# def single_profile2(variance_t, variance_est, region, figname) -> None:
#     pressure_levels = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]

#     fig, axs = plt.subplots(1, 1)
#     axs.plot((variance_t), pressure_levels, 'o', color='red', label="UM 'Truth'")
#     axs.plot((variance_est), pressure_levels, 'x', color='blue', label="MOGP Predicted")

#     axs.set_ylabel("Pressure (Pa)")
#     axs.set_xlabel("Standard Deviation")
#     axs.set_title(f"Standard Deviations - Region Number: {region}")
#     axs.legend()

#     plt.savefig(figname, format='png', bbox_inches='tight')
#     plt.close()


def plot_mogp_predictions(
    UM_t_var, 
    UM_q_var,
    MOGP_t_var, MOGP_t_unc,
    MOGP_q_var, MOGP_q_unc,
    region,
    output_path: str,
    sigma = 2,
) -> None:
    pressure_levels = [30, 100, 200, 300, 500, 700, 850, 925]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 8),
        sharey=True,
    )

    axes[0].set_title('Temperature / K')
    axes[0].plot((UM_t_var), np.flip(pressure_levels), 'x', color='red', label="UM 'Truth'")
    axes[0].errorbar((MOGP_t_var), np.flip(pressure_levels), xerr=MOGP_t_var+sigma*MOGP_t_unc, fmt='0')

    axes[1].set_title('Specific Humidity / kg/kg')
    axes[1].plot((UM_q_var), np.flip(pressure_levels), 'x', color='red', label="UM 'Truth'")
    axes[1].errorbar((MOGP_q_var), np.flip(pressure_levels), xerr=MOGP_q_var+sigma*MOGP_q_unc, fmt='0')

    axes.set_ylabel("Pressure (hPa)")
    axes.set_xlabel("Standard Deviation")
    fig.suptitle(f"Standard Deviations - Region Number: {region+1}")
    axes.legend()

    plt.savefig(
        os.path.join(output_path, f"mogp_pred_{(region+1):02d}.png")
    )
    plt.close()