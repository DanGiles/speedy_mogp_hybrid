Variance predictions of atmospheric variables

The steps in this guide should be followed in order.

## To set up the environment
- Install Python 3.10 (e.g. using [pyenv](https://github.com/pyenv/pyenv))
- Install [Poetry](https://python-poetry.org/)
- Run `poetry install`

# 1. Data Prep

## Prior to running data prep

Copy, edit and rename `data_prep/python/script_variables_template.py` to `data_prep/python/script_variables.py` such that the variables satisfy the needs of your setup. This must include root directories for reading and storing the data.

## Running data prep

Ensure python3 is ready and running the following command:

```
python loop_files.py <day>
```

where `<day>` is an integer. In this case, day takes values 1, 2, ..., 10.
`run_loop_files.sh` is already setup for run this.

## SPEEDY Run

### `run_first.sh`

Run the script `/speedy/model/run/run_first.sh`. This builds the SPEEDY executable and simulates a 12 month forecast: 1981/01/01-1982/01/01.

This script takes about 5 minutes to complete.

### `run_cycle.sh`

Run the script `/speedy/model/run/run_cylce.sh`. This script rebuilds the SPEEDY executable to run in 6 hourly chunks. This is necessary for our script later. `run_cycle.sh` also simulates a 10 year forecast which starts by using the final output from `run_first.sh`. 

This script takes about ?? to complete

# 2. SPEEDY Hybrid

## Prior to GP training or running SPEEDY Hybrid

Copy, edit and rename `hybrid/src/script_variables_template.py` to `hybrid/src/script_variables.py` such that the variables satisfy the needs of your setup. This must include root directories for reading and storing the data.

This file includes a flag for using a pre-trained GP model or training a new model when before launching SPEEDY. This default is to `True` i.e. train a new GP everytime.

## GP Training

The script `/hybrid/src/train-gp.sh` provides an example of a submission script for training the Guassian process. This example is designed for submission UCL's Myriad HPC. 

At the time of writing, the latest verion of the python package `mogp-emulator`, version 0.7.2, does not work for our needs. Instead, we revert to package version 0.6.1 and install directly from the [GitHub repository](https://github.com/alan-turing-institute/mogp-emulator).

Now turn the `TRAIN_GP` flag to `False` in your `script_variables.py` file. Future steps will load in the pre-trained Gaussian process.

## SPEEDY Hybrid Run

The script `/hybrid/src/run-speedy.sh` launches the `wrapper.py` script which will load the pre-trained Gaussian process and run SPEEDY. Each time step within the main loop will relaunch SPEEDY to run for a 6 hour forecast. This loop will produce 10 years worth of forecast where each 6 hour simulation represents a different potential atmosphere as sampled from the Gaussian process.

### SPEEDY Hybrid Notes

A few changes have been made to this code to suit our scientific needs.

Line 16 of `/speedy/model/source/makefile` has been commented out to ensure endianness of the programm remains consistent with the system default.


# 3. Analysis

See `analysis/READ_ME.md` for details on running the scripts which generate the figures in the manuscript.