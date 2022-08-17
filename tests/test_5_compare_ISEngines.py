#!/usr/bin/env python3
# coding: utf-8
#
# Test 5. Compare results of different ImageStructureEngines
#
# THIS SCRIPT SHOULD BE RUN FROM THE PROJECT ROOT DIRECTORY
#  (this is the parent directory to the directory in which this script is)
#  In order to run using 'Spyder', choose 'Run'->'Configuration per file', and
#  set the working directory to the project's root directory. Personally, I
#  also set 'Execute in a dedicated console' 
#
# Alternatively, to run using pytest (from the project root directory)
#
#   pytest tests/test_5_compare_ISEngines.py
#
#
#


import pytest

# import numpy as np
# import matplotlib.pyplot as plt

import ddm_toolkit
from ddm_toolkit.ddm import best_available_engine_model
from ddm_toolkit.ddm import available_engine_models
from ddm_toolkit.params import DDMParams
from ddm_toolkit.workflows import simul1_make_simulated_image_stack
from ddm_toolkit.workflows import compare_all_ImageStructureEngines



print('')
print('')
print('5. Test of different ImageStructureEngine models')
print('')
print('available models: ', available_engine_models)
print('best available engine: ', best_available_engine_model)
print('')



params = DDMParams() 

# set up simulation parameters
params.sim_Np = 100
params.sim_bl = 200.
params.sim_Nt = 300
params.sim_T = 600.
params.sim_D = 0.4
params.sim_img_border = 16.0
params.sim_img_w = 2.0
params.sim_img_Npx = 256
params.sim_img_I_offset = 0.06
params.sim_img_I_noise = 0.003
params.update_simulation_parameters()

# make a simulated image stack
print("creating simulated image stack")
ims = simul1_make_simulated_image_stack(params)
print("")

# set ISF and analysis  parameters
params.ISE_Npx = params.sim_img_Npx # use image width to set ISE width
params.ISE_Nbuf = 50
params.initial_guess_D = params.sim_D * 4.3 # choose an initial guess that is a factor 4.3 off

# run all available engine models
available_engine_models, dts, D_fits, allcloses = compare_all_ImageStructureEngines(ims, params)

print("")
print("Engine#    Time [s]  rel. speed       D_fit [m2 s-1]    np.allclose")
print("-------------------------------------------------------------------")
for i,ISEmodel in enumerate(available_engine_models):
    relspeed = dts[0]/dts[i]
    print('{0:4d}     {1:10.3f}  {2:10.3f}      {3:15.10f}      {4}'\
          .format(ISEmodel, dts[i], relspeed, D_fits[i], allcloses[i] ))

print('')
print('')

# tests for pytest
i_list = [i for i in range(len(available_engine_models))]

@pytest.mark.parametrize("i_model", i_list)
def test_ISEngine_run(i_model):
    assert allcloses[i_model]
    


