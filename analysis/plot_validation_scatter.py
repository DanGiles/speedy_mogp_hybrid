import os
import numpy as np
import matplotlib.pyplot as plt
from script_variables import *

ground_truth = np.load(os.path.join(gp_directory_root, "test_target.npy"))
predicted = np.load(os.path.join(gp_directory_root, "stds.npy"))
uncer = np.load(os.path.join(gp_directory_root, "uncer.npy"))

nlev = 8
vars = ['Temperature', 'Specific Humidity']
v = 0
for val, var in enumerate(vars):
    print(var)
    if val == 0:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (15,12))
        # plt.title(f"{var}")
        axs = axes.flatten()
        for i in range(nlev):
            axs[i].plot(ground_truth[v+i,:], ground_truth[v+i,:], '-')
            axs[i].errorbar(ground_truth[v+i,:], predicted[v+i,:], yerr=uncer[v+i, :], fmt='o')
            axs[i].set_title(f"Atmospheric Level {i}")
            if i == 0 or i == 4:
                axs[i].set_ylabel("Predicted std [K]")
            if i >= 4:
                axs[i].set_xlabel("Ground truth std [K]")
    else:
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 6)
        
        axs = axes.flatten()
        axes = fig.add_subplot(gs[0, 0:2])
        axes.plot(ground_truth[v,:], ground_truth[v,:], '-')
        axes.errorbar(ground_truth[v,:], predicted[v,:], yerr=uncer[v, :], fmt='o')
        axes.set_title(f"Atmospheric Level 0")
        axes.set_ylabel("Predicted std [g/kg]")


        axes = fig.add_subplot(gs[0, 2:4])
        axes.plot(ground_truth[v+1,:], ground_truth[v+1,:], '-')
        axes.errorbar(ground_truth[v+1,:], predicted[v+1,:], yerr=uncer[v+1, :], fmt='o')
        axes.set_title(f"Atmospheric Level 1")


        axes = fig.add_subplot(gs[0, 4:])
        axes.plot(ground_truth[v+2,:], ground_truth[v+2,:], '-')
        axes.errorbar(ground_truth[v+2,:], predicted[v+2,:], yerr=uncer[v+2, :], fmt='o')
        axes.set_title(f"Atmospheric Level 2")

        axes = fig.add_subplot(gs[1, 1:3])
        axes.plot(ground_truth[v+3,:], ground_truth[v+3,:], '-')
        axes.errorbar(ground_truth[v+3,:], predicted[v+3,:], yerr=uncer[v+3, :], fmt='o')
        axes.set_title(f"Atmospheric Level 3")
        axes.set_ylabel("Predicted std [g/kg]")
        axes.set_xlabel("Ground truth std [g/kg]")


        axes = fig.add_subplot(gs[1, 3:5])
        axes.plot(ground_truth[v+4,:], ground_truth[v+4,:], '-')
        axes.errorbar(ground_truth[v+4,:], predicted[v+4,:], yerr=uncer[v+4, :], fmt='o')
        axes.set_title(f"Atmospheric Level 4")
        axes.set_xlabel("Ground truth std [g/kg]")

        plt.subplots_adjust(wspace=0.5, hspace=0.2)

    v = v + 8
    plt.savefig(os.path.join(gp_directory_root, "validation", f"{var}_scatter_uncert.png"), dpi = 300,  bbox_inches='tight')
    plt.show()
