import os
import numpy as np
import matplotlib.pyplot as plt

from script_variables import *

# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/analysis' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs' #override for local compute, otherwise comment out

# Comment out variables to exclude
vars = {
    'precip': ['Precipitation', 'g/(m^2 s)', 'PREC'],
    # 'ps': ['Air pressure', 'Pa'], 
    # 'cloudc': ['Total cloud cover', 'fraction'], 
    # 'clstr': ['Stratiform cloud cover', 'fraction'], 
    # 'precnv': ['Convective precipitation', 'g/(m^2 s)'], 
    # 'precls': ['Large-scale precipitation', 'g/(m^2 s)'], 
    # 'tsr': ['Top-of-atm. Shortwave radiation', 'downward W/m^2'], 
    'olr': ['Outgoing longwave radiation', 'upward W/m^2', 'OLR'], 
    # 'u': ['Wind speed (u)', 'm/s'], 
    # 'v': ['Wind speed (v)', 'm/s'], 
    # 't': ['Temperature', 'K'], 
    # 'q': ['Specific humidity', 'Kg/Kg'],
}
seasons = ['DJF', 'JJA']

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

def plot(ax, x, y, marker, label) -> None:
    ax.scatter(x, y, marker=marker, label=label)
    ax.plot(x, y)

for season in seasons:
    for var, info in vars.items():
        satellite = np.loadtxt(os.path.join(analysis_root, 'climate_data', f'{info[2]}_{season}.csv'), delimiter=',', dtype='float', skiprows=1)

        SPEEDY = np.load(os.path.join(analysis_root, GP_name, f"mean_{var}_{season}.npy"))
        SPEEDY_mean = np.mean(SPEEDY, axis=0)
        HYBRID = np.load(os.path.join(analysis_root, GP_name, f"mean_{var}_{season}.npy"))
        HYBRID_mean = np.mean(HYBRID, axis=0)

        x = np.linspace(-90, 90, num=len(SPEEDY_mean))

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )
        ax.grid()

        plot(ax, x, SPEEDY_mean, 'o', 'Speedy')
        plot(ax, x, HYBRID_mean, '*', 'Hybrid')
        plot(ax, satellite[:, 0], satellite[:, 1], 'x', 'Satellite')

        ax.set_xlabel('Latitude')
        ax.legend()
        fig.suptitle(f'Zonal {info[0]} [{info[1]}] - {season}')

        plt.savefig(
            os.path.join(output_path, f'zonal_{var}_{season}.png')
        )
        plt.close()