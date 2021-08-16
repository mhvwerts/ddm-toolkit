#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# The DDM Team, 2020-2021
#
# diffusion coefficient in => diffusion coefficient out
#
# STEP 3: Calculate Image Structure Function from synthetic video
#
# Usage:
#        python3 simul3_ISF_from_simulation.py [<name of configuration file>]
#


import numpy as np
import matplotlib.pyplot as plt


from ddm_toolkit import DDMParams_from_configfile_or_defaultpars
from ddm_toolkit.workflows import simul2_load_simulation_result_file
from ddm_toolkit.workflows import simul3_calculate_ISF


if __name__ == "__main__":
    #######################
    # This script is designed to run from the command line or via Spyder IDE
    #
    # Not for use in a Notebook
    ########################
    
    # Command line interface
    params = DDMParams_from_configfile_or_defaultpars()
    simulfpn = params.vidfpn

    # LOAD data (and parameters stored inside the file, but these are not used)
    ims, params_simulfile = simul2_load_simulation_result_file(simulfpn)
       
    # calculate image structure function
    IA = simul3_calculate_ISF(ims, params)
    
    # write result file for next step
    print('Writing result to file', params.ISE_outfpn)
    IA.save(params.ISE_outfpn)

    #
    # quick plot of radially average image structure function
    #
    IAqtau = np.zeros((len(IA.tauf),len(IA.u)))
    for i in range(len(IA.tauf)):
        IAqtau[i,:] = IA.radavg(i)
        
    plt.figure("radially averaged (video) image structure function")
    plt.imshow(IAqtau)
    print("Close open graph windows to terminate script")
    plt.show()











