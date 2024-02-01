import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from typing import List, Tuple

from script_variables import *

# SPEEDY_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/speedy' #override for local compute, otherwise comment out
# analysis_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/from_dan' #override for local compute, otherwise comment out
# pngs_root = '/Users/jamesbriant/Documents/Projects/ml_climate_fusion/pngs/pnas' #override for local compute, otherwise comment out

#################
# See the bottom of this script for plotting
#################

output_path = os.path.join(pngs_root, GP_name)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'lifted_index')
if not os.path.isdir(output_path):
    os.mkdir(output_path)


def round_nearest_half(x):
    return round(x * 2.0)/2

def plot(
    locations: str | List[str], 
    points: List[int], 
    season: str, 
    nrows: int, 
    ncols: int, 
    figsize: Tuple[int, int] =(8, 8),
    filename: str = None
):
    if len(points) != nrows*ncols:
        raise ValueError("Number of points must equal nrows*ncols")
    if isinstance(locations, str):
        locations = [locations]*len(points)
    if len(locations) != len(points):
        raise ValueError("Number of locations must equal number of points")

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        figsize=figsize,
        sharex=True,
        sharey=True
    )
    # fig.suptitle(f'Lifted Index Histogram \n Season: {season} - Regions: (top) {"".join([f"{locations[i].capitalize()}: {points[i]+1}, " for i in range(len(points))])} (bottom)')
    fig.suptitle(f'Lifted Index Histogram \n Season: {season}')
    for i, point in enumerate(points):
        location = locations[i]

        speedy = np.load(os.path.join(analysis_root, 'speedy_seasonal', f"{location}_lifted_index_{season}.npy"))
        hybrid = np.load(os.path.join(analysis_root, 'hybrid_seasonal', f"{location}_lifted_index_{season}.npy"))

        LI_SPEEDY = speedy[point, :]
        LI_HYBRID = hybrid[point, :]

        # Count and print the number of missing points. Recall default value was set to 9999
        if np.max(LI_SPEEDY)>9000:
            # print(f'{season} - {location}, point {point+1} - SPEEDY: {np.sum(LI_SPEEDY>9000)}')
            LI_SPEEDY = LI_SPEEDY[LI_SPEEDY < 9000]
        
        if np.max(LI_HYBRID)>9000:
            # print(f'{season} - {location}, point {point+1} - Hybrid: {np.sum(LI_HYBRID>9000)}')
            LI_HYBRID = LI_HYBRID[LI_HYBRID < 9000]

        bin_min = round_nearest_half(min(np.min(LI_SPEEDY), np.min(LI_HYBRID)))
        bin_max = round_nearest_half(max(np.max(LI_SPEEDY), np.max(LI_HYBRID)))
        bin_points = np.arange(bin_min, bin_max, 0.5)

        ############### PLOTTING ###############
        y1, x1, _ = axes.flatten()[i].hist(LI_SPEEDY, bins=bin_points, alpha=0.5, label="SPEEDY", color="g")
        y2, x2, _ = axes.flatten()[i].hist(LI_HYBRID, bins=bin_points, alpha=0.5, label="Hybrid", color="m")

        y = max(max(y1), max(y2)) + 50

        axes.flatten()[i].vlines(np.mean(LI_SPEEDY), 0, y, colors='g', linestyles='dashed')
        axes.flatten()[i].vlines(np.mean(LI_HYBRID), 0, y, colors='m', linestyles='dashed')

        u = [np.mean(LI_SPEEDY), np.var(LI_SPEEDY), skew(LI_SPEEDY), kurtosis(LI_SPEEDY)]
        v = [np.mean(LI_HYBRID), np.var(LI_HYBRID), skew(LI_HYBRID), kurtosis(LI_HYBRID)]

        part1 = f'SPEEDY - mean: {u[0]:.3f}; var: {u[1]:.3f}; skew: {u[2]:.3f}; kurt: {u[3]:.3f}'
        part2 = f'Hybrid - mean: {v[0]:.3f}; var: {v[1]:.3f}; skew: {v[2]:.3f}; kurt: {v[3]:.3f}'

        axes.flatten()[i].set_title(f'{part1} \n {part2}')
        axes.flatten()[i].legend()
        axes.flatten()[i].text(0.05, 0.8, f'{locations[i].capitalize()}\nPoint: {point+1}', transform = axes.flatten()[i].transAxes)

    if filename is not None:
        plt.savefig(
            os.path.join(output_path, f'{filename}.png')
        )
    # plt.savefig(
    #     os.path.join(output_path, f'{location}_{point+1}_{season}_lifted_index.png')
    # )
    # plt.close()

    plt.show()




# Africa point 21 (22 on map) sits around Uganda? It shows the biggest blue blob in Africa.
# Arabia point 14 (15 on map)
# India point 12 (13 on map) is over the Indian continent.
# India point 31 (32 on map) is over the Bay of Bengal halfway between Sri Lanka and northern tip of Simatra, Indonesia.

plot(
    ['africa', 'india'], 
    [21, 12], 
    'DJF', 
    nrows=2, ncols=1
)
plot(
    ['africa', 'arabia', 'india', 'india'], 
    [21, 14, 12, 31], 
    'DJF', 
    nrows=4, 
    ncols=1, 
    figsize=(8, 13),
    filename='lifted_index_comparison_4x1'
)
plot(
    ['africa', 'arabia', 'india', 'india'], 
    [21, 14, 12, 31], 
    'DJF', 
    nrows=2, 
    ncols=2, 
    figsize=(13, 7.5),
    filename='lifted_index_comparison_2x2'
)