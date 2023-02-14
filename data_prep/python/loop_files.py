#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os

def read_data(file, t, oper, code):
    file_str = file + t + '_'+ oper + '_'+ code +'.nc'
    data = Dataset(file_str)
    values = data.variables['unknown']
    return values

def loop(codes, base_file, timestamps, oper, input_array):
    rootdir = ''
    lead = '20200101T0000Z_'
    after = '_km3p3_RA2M_32x32sampling_hence_14x14_time'
    site = 0
    for i in range(len(base_file)):
        region = base_file[i].split()
        j = 0
        for stash in codes:
            file_name = rootdir + region[0] + '/' + lead + region[0] + after
            tim = 0
            for t in timestamps:
                data = read_data(file_name, t, oper, stash)
                data = np.reshape(data, (70,196))
                input_array[j,:,:,site,tim] = data
                tim = tim + 1
            j = j + 1
        site = site + 1
    return

def loop_2d(codes,  base_file, timestamps, oper, input_array):
    rootdir = ''
    lead = '20200101T0000Z_'
    after = '_km3p3_RA2M_32x32sampling_hence_14x14_time'
    site = 0 
    for i in range(len(base_file)):
        region = base_file[i].split()
        j = 0
        for o in oper:
            file_name = rootdir + region[0] + '/' + lead + region[0] + after
            data = read_data(file_name, timestamps[0], o, codes[0])
            data = np.array(data)
            input_array[j,:,site] = data.flatten()
            j = j + 1
        site = site + 1
    return


with open('sites.txt','r') as file:
    txt = file.readlines()
    stash = ['00408', '16004','00010']
    time = ['000', '018', '036', '054']
    oper = 'AVG'
    input_array = np.zeros((3, 70, 196,len(txt), len(time)))
    loop(stash, txt, time, oper, input_array)
    np.save('processed/mean.npy', input_array)
    oper = 'STD'
    input_array = np.zeros((3, 70, 196, len(txt), len(time)))
    loop(stash, txt, time, oper, input_array)
    np.save('processed/std.npy', input_array)
    
    twodstash = ['00033']
    oper = ['AVG', 'STD']
    input_array = np.zeros((2, 196, len(txt)))
    loop_2d(twodstash, txt, time, oper, input_array)
    np.save('processed/orography.npy', input_array)

    twodstash = ['00030']
    oper = ['AVG', 'STD']
    input_array = np.zeros((2, 196, len(txt)))
    loop_2d(twodstash, txt, time, oper, input_array)
    np.save('processed/land_sea.npy', input_array)

file.close() 


#np.zeros((3,70,(196*len(base_file))))
#loop(stash, base_file, time, oper, input_array)
#np.save('../processed/std.npy', input_array)

# Two Dimensional Data
# twodstash = ['00033']#, '00033']
# oper = ['AVG', 'STD']
# input_array = np.zeros((2,(196*len(base_file))))
# loop_2d(twodstash, base_file, time, oper, input_array)
# np.save('../processed/orography.npy', input_array)
