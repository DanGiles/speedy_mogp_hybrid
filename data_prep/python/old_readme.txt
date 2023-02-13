For use by Daniel Giles at University College London.
Data produced by Cyril Morcrette at the Met Office (2021).

The directory structure is as follows:

moose:adhoc/projects/ml_moments/yyyymmdd/r??/

with a different doirectory for each day and then for each of the 99 limited-area models (r01 to r99).

Then in each directory there are 1464 files

These have filenames like:

20200101T0000Z_r01_km3p3_RA2M_32x32sampling_hence_14x14_time069_STD_16004.nc
             1   2     3    4     5                   6       7   8     9
1 date/time
2 which of the 99 limited area models this file is for
3 the grid-length of the model used to produce the data km3p3 means dx = 3.3 km
4 the configuration used for the model (RA2M is what has been used operationally over the UK)
5 size of region that data is "processed" over. 32x32 means (along with dx=3.3 means we are processing over 105.6km typical of a climate model grid-box). 
6 Since each limited rea model is run over a large area, despite doing 32x32 averaging, we can do that 14x14 times.
7 time since the date/time from [1] in steps of 20 minutes. So time000 = 00:00 midnight, and time069 is 23:40
8 processing that has been done 
  AVG=mean, STD=standard deviation, SKW=skewness, COR is correlation coefficient, 
  BCU_???_PPN are *estimates* of how much of the precipitation (that is treated explicitly in this high resolution model, 
  can be interpreted as being from Buoyant Cloudy Updrafts (with ??? being the threshold for "buoyant", "cloudy" and "updraft" respectively) 
  and hence could be expected to be associated with the convection scheme if this data column were to be processes in a coarse-resolution model with a convection scheme.
9 see below:
  # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 4 list_* variables.
  # ------------------------------
  # 2d fields (these are 2d fields that do NOT vary with time).
  #  0  30 Land-sea mask
  #  0  33 Orography
  # ------------------------------
  # 4d fields (i.e. 3d fields that vary with time)
  #  0   2 u wind      (on rho levels and staggered grid)
  #  0   3 v wind      (on rho levels and staggered grid)
  #  0 150 w wind      (on theta levels)
  #  0  10 qv          (on theta levels)
  #  0  12 qcf         (on theta levels)
  #  0 254 qcl         (on theta levels)
  #  0 272 qrain       (on theta levels)
  #  0 273 qgraupel    (on theta levels)
  #  0   4 theta       (on theta levels)
  #  0 408 pressure    (on theta levels)
  # 16   4 temperature (on theta levels)
  # ------------------------------
  # 3d fields (these 2d fields are technically 3d because of time dimension)
  #  1 207 Incoming SW Rad Flux (TOA)
  #  3 217 Surface Sensible Heat Flux
  #  3 234 Surface Latent Heat Flux
  # 16 222 Mean-Sea-Level Pressure
  # ------------------------------

Each file is then a simple netcdf file (not loads of metadata).

Data has been produced by suite:
u-cg081 for 1-5 Jan 2010
u-cg481 for 6-17 Jan 2020.

Data retrieved and processed using:
process_ml_lams.m
run_retrieve_and_process_ml_lams_on_spice.sh
retrieve_and_process_ml_lams_MASTERCOPY.py
(python code submitted onto spice, via a matlab meta script).
