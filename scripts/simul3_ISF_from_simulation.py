#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# diffusion coefficient in => diffusion coefficient out
#
# STEP 3: Calculate Image Structure Function from synthetic video

# The DDM Team, 2020-2021
#
# Usage:
#        python3 simul3_ISF_from_simulation.py [<name of configuration file>]
#

#%% Insert path to import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
from ddm_toolkit.params import DDMParams_from_configfile_or_defaultpars

#%%
from ddm_toolkit.workflows import simul2_load_simulation_result_file
from ddm_toolkit.workflows import simul3_calculate_ISF

#%%
if __name__ == "__main__":

    
    # Get simulation/analysis parameters
    params = DDMParams_from_configfile_or_defaultpars()
    simulfpn = params.vidfpn

    # LOAD data (and parameters stored inside the file, but these are not used)
    ims, params_simulfile = simul2_load_simulation_result_file(simulfpn)
       
    # calculate image structure function
    IA = simul3_calculate_ISF(ims, params)
    
    # write result file for next step
    print('Writing result to file', params.ISF_outfpn)
    IA.save(params.ISF_outfpn)

    
    
    











