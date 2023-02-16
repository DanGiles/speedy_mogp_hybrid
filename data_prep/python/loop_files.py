#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt
from netCDF4 import Dataset
# import os
import sys

from typing import List

#script below contains information regarding the project set.
#Edit script_variables.py before running loop_files.py
from script_variables import *


def read_data(
    filename: str, 
    hour: str, 
    oper: str, 
    code: str
) -> np.ndarray:
    full_filename = f"{filename}{hour}_{oper}_{code}.nc"
    data = Dataset(full_filename)
    values = data.variables['unknown']
    return values


def loop(
    day: int, 
    stash_codes: List[str], 
    # base_file, 
    timestamps: List[str], 
    operation: str, 
    input_array: np.ndarray
) -> None:
    date = f'202001{day:02d}'
    for i in range(subregion_count):
        subregion = i+1
        for j, stash_code in enumerate(stash_codes):
            filename = f"{input_root}/{date}/{date}T0000Z_r{subregion:02d}_km1p5_RA2T_224x224sampling_hence_2x2_time"
            for t, timestamp in enumerate(timestamps):
                data = read_data(
                    filename,
                    timestamp,
                    operation,
                    stash_code
                )
                data = np.reshape(data, (UM_levels, subregion_count))
                input_array[j, :, :, i, t] = data


# def loop(codes, base_file, timestamps, oper, input_array):
#     rootdir = ''
#     lead = '20200101T0000Z_'
#     after = '_km3p3_RA2M_32x32sampling_hence_14x14_time'
#     site = 0
#     for i in range(len(base_file)):
#         region = base_file[i].split()
#         j = 0
#         for stash in codes:
#             file_name = rootdir + region[0] + '/' + lead + region[0] + after
#             tim = 0
#             for t in timestamps:
#                 data = read_data(file_name, t, oper, stash)
#                 data = np.reshape(data, (70,196))
#                 input_array[j,:,:,site,tim] = data
#                 tim = tim + 1
#             j = j + 1
#         site = site + 1
#     return

def loop_2d(
    day: int, 
    stash_code: str, 
    # base_file, 
    timestamp: str, 
    operations: List[str], 
    input_array: np.ndarray
) -> None:
    date = f'202001{day:02d}'
    for i in range(subregion_count):
        subregion = i+1
        for k, operation in enumerate(operations):
            filename = f"{input_root}/{date}/{date}T0000Z_r{subregion:02d}_km1p5_RA2T_224x224sampling_hence_2x2_time"
            data = read_data(
                filename,
                timestamp,
                operation,
                stash_code
            )
            input_array[k, :, i] = np.array(data).flatten()

# def loop_2d(codes,  base_file, timestamps, oper, input_array):
#     rootdir = ''
#     lead = '20200101T0000Z_'
#     after = '_km3p3_RA2M_32x32sampling_hence_14x14_time'
#     site = 0 
#     for i in range(len(base_file)):
#         region = base_file[i].split()
#         j = 0
#         for o in oper:
#             file_name = rootdir + region[0] + '/' + lead + region[0] + after
#             data = read_data(file_name, timestamps[0], o, codes[0])
#             data = np.array(data)
#             input_array[j,:,site] = data.flatten()
#             j = j + 1
#         site = site + 1
#     return


def main(day: int):
    
    stash = ['00408', '16004','00010']
    time = ['000', '006', '012', '018']
    oper = 'AVG'
    input_array = np.zeros(
        (3, UM_levels, subregion_count, region_count, len(time))
    )
    loop(day, stash, time, oper, input_array)
    np.save(f'202001{day:02d}_mean.npy', input_array)

    ### THESE DIMENSIONS NEED CHANGING
    stash = ['16004','00010']
    oper = 'STD'
    input_array = np.zeros(
        (2, UM_levels, subregion_count, region_count, len(time))
    )
    loop(day, stash, time, oper, input_array)
    np.save(f'202001{day:02d}_std.npy', input_array)
    

    twodstash = ['00033']
    oper = ['AVG', 'STD']
    input_array = np.zeros(
        (2, subregion_count, region_count)
    )
    loop_2d(day, twodstash, time, oper, input_array)
    np.save(f'202001{day:02d}_orography.npy', input_array)

    ### THESE DIMENSIONS NEED CHANGING
    twodstash = ['00030']
    oper = ['AVG']
    input_array = np.zeros(
        (1, subregion_count, region_count)
    )
    loop_2d(day, twodstash, time, oper, input_array)
    input_array.squeeze(axis=0) #this line removes the first dimension.
    np.save(f'202001{day:02d}_land_sea.npy', input_array)

    # file.close() #I don't this this line does anything?


if __name__ == "__main__":
    day = int(sys.argv[1])
    main(day)

#np.zeros((3,70,(196*len(base_file))))
#loop(stash, base_file, time, oper, input_array)
#np.save('../processed/std.npy', input_array)

# Two Dimensional Data
# twodstash = ['00033']#, '00033']
# oper = ['AVG', 'STD']
# input_array = np.zeros((2,(196*len(base_file))))
# loop_2d(twodstash, base_file, time, oper, input_array)
# np.save('../processed/orography.npy', input_array)
