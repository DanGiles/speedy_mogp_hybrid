#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from typing import List

import numpy as np
from netCDF4 import Dataset

#Script below contains information regarding the project dataset and directories.
#Edit script_variables.py before running loop_files.py - See README.md
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
    timestamps: List[str], 
    operation: str, 
    output_array: np.ndarray
) -> None:
    date = f'202001{day:02d}'
    for region in range(region_count):
        for j, stash_code in enumerate(stash_codes):
            
            filename = f"{input_root}/{date}/{date}T0000Z_r{(region)+1:02d}_km1p5_RA2T_224x224sampling_hence_2x2_time"
            for t, timestamp in enumerate(timestamps):
                data = read_data(
                    filename,
                    timestamp,
                    operation,
                    stash_code
                )
                data = np.reshape(data, (UM_levels, subregion_count))
                for subregion in range(subregion_count):
                    output_array[j, :, region, subregion,  t] = data[:,subregion]

    output_array = np.reshape(output_array, (len(stash_codes), UM_levels, (region_count*subregion_count), len(timestamps)))
    return output_array


def loop_2d(
    day: int, 
    stash_code: str, 
    timestamp: str, 
    operations: List[str], 
    output_array: np.ndarray
) -> None:
    date = f'202001{day:02d}'
    for region in range(region_count):
        for j, operation in enumerate(operations):
            filename = f"{input_root}/{date}/{date}T0000Z_r{(region+1):02d}_km1p5_RA2T_224x224sampling_hence_2x2_time"
            data = read_data(
                filename,
                timestamp,
                operation,
                stash_code
            )
            data = np.reshape(data, (subregion_count))
            for subregion in range(subregion_count):
                output_array[j, region, subregion] = data[subregion]
    output_array = np.reshape(output_array, (len(operations), (region_count*subregion_count)))
    return output_array

def main(day: int):
    
    stash = ['00408', '16004','00010']
    time = ['000', '006', '012', '018']
    oper = 'AVG'
    input_array = np.zeros(
        (3, UM_levels, region_count, subregion_count, len(time))
    )
    input_array = loop(day, stash, time, oper, input_array)
    np.save(f'{output_root}/202001{day:02d}_mean.npy', input_array)

    stash = ['16004','00010']
    oper = 'STD'
    input_array = np.zeros(
        (2, UM_levels, region_count, subregion_count, len(time))
    )
    input_array = loop(day, stash, time, oper, input_array)
    np.save(f'{output_root}/202001{day:02d}_std.npy', input_array)
    

    twodstash = ['00033']
    oper = ['AVG', 'STD']
    input_array = np.zeros(
        (2, region_count, subregion_count)
    )
    input_array = loop_2d(day, twodstash[0], time[0], oper, input_array)
    np.save(f'{output_root}/202001{day:02d}_orography.npy', input_array)

    twodstash = ['00030']
    oper = ['AVG']
    input_array = np.zeros(
        (1, region_count, subregion_count)
    )
    input_array = loop_2d(day, twodstash[0], time[0], oper, input_array)
    input_array = np.squeeze(input_array, axis=0) #this line removes the first dimension.
    np.save(f'{output_root}/202001{day:02d}_land_sea.npy', input_array)



if __name__ == "__main__":
    day = int(sys.argv[1])
    main(day)
