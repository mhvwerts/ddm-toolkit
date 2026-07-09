#!/usr/bin/env python3
# coding: utf-8
#
# Test 5. Compare results of different ImageStructureEngines
#
#

#
#
import pytest



# configure test technical details
warmup = 2 


# import numpy as np
# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

import ddm_toolkit

from ddm_toolkit.ddm import best_available_engine_model
from ddm_toolkit.ddm import available_engine_models
from ddm_toolkit.params import DDMParams
from ddm_toolkit.workflows import simul1_make_simulated_image_stack
from ddm_toolkit.ddm import ImageStructureEngine
from ddm_toolkit.ddm import ImageStructureFunction
from ddm_toolkit.analysis import ISFanalysis_simple_brownian

from ddm_toolkit.ddm_new import DifferentialImageCorrelator
from ddm_toolkit.ddm_new import DifferentialImageCorrelationFunction
from ddm_toolkit.analysis_new import DICFanalysis_simple_brownian


## Global data

ISFref = None # reserve storage



##  routines

def compare_all_ImageStructureEngines(ims, params):
    """
    Compare the results of all available ImageStructureEngines

    Parameters
    ----------
    ims : list of 2D np.arrays (or a 3D np.array)
        Stack of video frames.
    params : DDMParams
        Object containing DDM Toolkit parameters.

    Returns
    -------
    result tuple : (available_engine_models, dts, D_fits, allcloses)
    """

    global ISFref

    dts = []
    D_fits = []
    allcloses = []
    for i,ISEmodel in enumerate(available_engine_models):
        print('Engine model #', ISEmodel)
        params.ISE_type = ISEmodel

        #####
        ##### DDM: calculate Image Structure Function
        #####

        ISE1 = ImageStructureEngine(params.ISE_Npx, params.ISE_Nbuf,
                                    engine_model = params.ISE_type)
    
        # fill buffer (plus some warm-up)
        for it in tqdm(range(0, params.ISE_Nbuf+warmup)):
            ISE1.push(ims[it])
       
        # do heavy lifting (here, we can measure performance!)
        tstart = timer()
        for it in tqdm(range(params.ISE_Nbuf+warmup, params.Nframes)):
            ISE1.push(ims[it])
        dt = timer() - tstart

        # output Image Structure Function        
        ISF = ImageStructureFunction.fromImageStructureEngine(ISE1)
        # set real world units
        ISF.real_world(params.um_p_pix, params.s_p_frame)
       
        #####
        ##### 

        ##### Brownian analysis
        result = ISFanalysis_simple_brownian(ISF, params.initial_guess_D)
        dts.append(dt)
        D_fits.append(result.D_fit)
        print('D_fit = ',result.D_fit)
        if i==0:
            ISFref = ISF.ISF.copy()
            allcloses.append(True)
        else:
            allcloses.append(np.allclose(ISF.ISF, ISFref,
                                        rtol=5e-05, atol=1e-08))
        print('')
    return(available_engine_models, dts, D_fits, allcloses)



def ddm2025engine(params):
    # we insert the new DICF engine here
    # accumulate differential image correlation function
    
    print("Engine model 2025")
    Nbuf = params.ISE_Nbuf
    Npx = params.ISE_Npx
    
    #####
    ##### DDM: calculate Differential Image Correlation Function
    #####
    
    # Ntau_req = 100 
    # DICFe = DICF_engine.log_tau(Npx, Nbuf, Ntau_req)
    DICFe = DifferentialImageCorrelator.lin_tau(Npx, Nbuf)
    
    # number of 'warm-up' heavy lifts before performance measurement
    # allow, for example, the first Numba compile
    
    # fill buffer (plus some warm-up)
    for it in tqdm(range(0, DICFe.Nbuf+warmup)):
        DICFe.push(ims[it])
        
    # do heavy lifting (here, we can measure performance!)
    tstart = timer()
    for it in tqdm(range(DICFe.Nbuf+warmup, params.Nframes)):
        DICFe.push(ims[it])
    dt = timer() - tstart    
    
    print()
    
    DICF1 = DifferentialImageCorrelationFunction\
            .fromDifferentialImageCorrelator(DICFe)
    DICF1.real_world(params.um_p_pix, params.s_p_frame)
          
    #####
    ##### 

    ##### Brownian analysis
    result = DICFanalysis_simple_brownian(DICF1, params.initial_guess_D)
    dts.append(dt)
    # D_fits.append(result.D_fit)
    print('D_fit = ',result.D_fit)
    print()
    
    return dt, result.D_fit, DICF1.values




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
params.ISE_Nbuf = 30
params.initial_guess_D = params.sim_D * 4.3 # choose an initial guess that is a factor 4.3 off

###
# run available "historic" engine models
available_engine_models, dts, D_fits, allcloses = compare_all_ImageStructureEngines(ims, params)

###
# run 2025 engine & compare to historic DICF/ISF (same criteria as others)
dt, Dfit, ddm2025DICF = ddm2025engine(params)
available_engine_models.append(2025)
dts.append(dt), 
D_fits.append(Dfit)
allcloses.append(np.allclose(ddm2025DICF, ISFref[1:],rtol=5e-05, atol=1e-08))



#### Generate result table

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
    


