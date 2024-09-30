#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import numpy.ma as ma
from script_variables import *

def read_const_grd(filename, nlon, nlat):
    num = 12
    f = np.fromfile(filename, dtype=np.float32)
    shape = (nlon,nlat,num)
    data = np.reshape(f, shape, order="F")
    data = data.astype(np.float64)
    return data

def write_fortran(filename, data):
    f=open(filename,'wb+')
    # data = data.flatten()
    data = data.astype(np.float32)
    fortran_data=np.asfortranarray(data,'float32')
    fortran_data.T.tofile(f)
    f.close()
    return


nlon = 96
nlat = 48

sst = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sst.grd"), nlon, nlat)
print(sst[:,:,0])
data = np.zeros_like(sst)
for year in range(12):
    z = ma.masked_less_equal(sst[:,:,year], -999.)
    data[:,:,year] = z + 4
print(data[:,:,0])

print(np.max(data-sst), np.min(data-sst))

write_fortran(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "sst_4K.grd"), data)

stl = read_const_grd(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "stl.grd"), nlon, nlat)
print(stl[:,:,0])
data = np.zeros_like(stl)
for year in range(12):
    z = ma.masked_less_equal(stl[:,:,year], -999.)
    data[:,:,year] = z + 4
print(data[:,:,0])

print(np.max(data-stl), np.min(data-stl))
write_fortran(os.path.join(SPEEDY_root, "model", "data/bc/t30/clim", "stl_4K.grd"), data)