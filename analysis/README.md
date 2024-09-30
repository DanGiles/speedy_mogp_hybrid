# Analysis scripts

This directory contains the scripts which handle the simulation output data and generate the figures used in the manuscript.
The subdirectory `analysis/auxiliary/` contains (some) of the scripts used to generate figures which do not appear in the manuscript but helped us to reach our conclusions.
The subdirectory contains another `README.md` file which you should read if you wish to run those scripts.

## STEP 0

As with the data_prep and hybrid directories, please copy and rename the `script_variables_template.py` from THIS directory and complete the required paths.

## STEP 1

**If** you ran the `hybrid/src/postprocess.py` script after running the hybrid simulation the proceed directly to step 2. 

Otherwise... Run:
1. `create_nc_files.py`,
2. `create_nc_fluxes_files.py`.

## STEP 2

Run `create_lifted_index_from_nc.py`.

## STEP 3

Run any of the scripts beginning with `plot_[...].py`.

- Figure 3 -> `plot_std_precip.py`
- Figure 4 -> `plot_annual_field_diff.py`
- Figure 5 -> `plot_monthly_means.py`
- Figure 6-right -> `plot_lifted_index_precipitation.py`
- Figure 7 -> `plot_validation_scatter.py`
- Figure 8 -> `plot_validation_scatter.py`

Figure 5 needs ERA5 data. Dan to provide the script for generating the ERA5 .nc files.

Figures 1, 2 and 6-left cannot be generated using these scripts. Please contact cyril.morcrette@metoffice.gov.uk regarding these figures.