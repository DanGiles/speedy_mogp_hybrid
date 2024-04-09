import os
import numpy as np
import matplotlib.pyplot as plt
from script_variables import *

ground_truth = np.load(os.path.join(neutral_root, "validation", "test_target.npy"))
predicted = np.load(os.path.join(neutral_root, "validation", "stds.npy"))
uncer = np.load(os.path.join(neutral_root, "validation", "uncer.npy"))

nlev = 8
vars = ['Temperature', 'Specific Humidity']
v = 0
for var in vars:
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (15,12))
    plt.suptitle(f"{var}")
    axs = axes.flatten()
    for i in range(nlev):
        # axs[i].plot(ground_truth[v+i,:], predicted[v+i,:], '*')
        axs[i].plot(ground_truth[v+i,:], ground_truth[v+i,:], '-')
        axs[i].errorbar(ground_truth[v+i,:], predicted[v+i,:], yerr=uncer[v+i, :], fmt='o')
        axs[i].set_title(f"Atmospheric Level {i}")
        if i == 0 or i == 4:
            axs[i].set_ylabel("Predicted")
        if i >= 4:
            axs[i].set_xlabel("Ground Truth")
    v = v + 8
    plt.savefig(os.path.join(neutral_root, "validation", f"{var}_scatter_uncert.png"), dpi = 300,  bbox_inches='tight')
    plt.show()

