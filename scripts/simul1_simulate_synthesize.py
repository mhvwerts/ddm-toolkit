#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# STEP 1: Brownian simulation and Video synthesis
#    (New version, based on "workflows")
#
# The DDM Team, 2020-2022
#
# Usage:
#        python3 simul1_simulate_synthesize.py [<name of parameter file>]
#

#%% Insert path to import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

#%%
from ddm_toolkit.params import DDMParams_from_configfile_or_defaultpars

#%% 
from ddm_toolkit.workflows import simul1_make_simulated_image_stack
from ddm_toolkit.workflows import simul1_save_simulation_result_file

#%%
if __name__ == "__main__":
     
    # Get simulation/analysis parameters
    params = DDMParams_from_configfile_or_defaultpars()
    
    # TO DO : Write a file parameter ? 
    
    # Simulate an image stack
    ims = simul1_make_simulated_image_stack(params)
    
    
    # Save simulated image stack (i.e video) and parameter used
    print('Writing result to file', params.vidfpn)
    simul1_save_simulation_result_file(params.vidfpn,
                                      ims, params)

    
