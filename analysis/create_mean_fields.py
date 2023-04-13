#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np

from script_variables import *

vars_2d = ['precip', 'p', 'cloudc', 'clstr', 'precnv', 'precls', 'tsr', 'olr']
vars_3d = ['u', 'v', 't', 'q']


for season in ['DJF', 'JJA']:
    for run in ['nature', 'fusion']:
        for var in vars_2d + vars_3d:
            f = np.load(os.path.join(analysis_root, f"{var}_{season}_{run}.npy"))

            # calculate mean
            output_mean = np.mean(f, axis=-1)
            np.save(
                os.path.join(analysis_root, f"mean_{var}_{season}_{run}.npy"),
                output_mean
            )

            # calculate variance
            output_var = np.var(f, axis=-1)
            np.save(
                os.path.join(analysis_root, f"var_{var}_{season}_{run}.npy"),
                output_var
            )