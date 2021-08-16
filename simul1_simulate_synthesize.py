#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# STEP 1: Brownian simulation and Video synthesis
#    (New version, based on "workflows")
#
# The DDM Team, 2020-2021
#
# Usage:
#        python3 simul1_simulate_synthesize.py [<name of parameter file>]
#



from ddm_toolkit import DDMParams_from_configfile_or_defaultpars
from ddm_toolkit.workflows import simul1_make_simulated_image_stack
from ddm_toolkit.workflows import simul1_save_simulation_result_file

if __name__ == "__main__":
    # 
    # GET SIMULATION/ANALYSIS PARAMETERS
    #
    params = DDMParams_from_configfile_or_defaultpars()


    # make simulated image stack
    ims = simul1_make_simulated_image_stack(params)

    #
    # save video and parameter set used
    #
    print('Writing result to file', params.vidfpn)
    simul1_save_simulation_result_file(params.vidfpn,
                                       ims, params)

    
