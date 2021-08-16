#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:34:35 2021

@author: werts
"""

from timeit import default_timer as timer

import numpy as np

from ddm_toolkit import tqdm
from ddm_toolkit.simulation import ParticleSim2DBrownian
from ddm_toolkit.simulation import ImageSynthesizer2D
from ddm_toolkit.analysis import ISFanalysis_simple_brownian

from ddm_toolkit import ImageStructureEngine
from ddm_toolkit import ImageStructureFunction
from ddm_toolkit.ddm import available_engine_models


def simul1_make_simulated_image_stack(dparams):
    """
    Do a basic Brownian simulation and generate corresponding synthetic video

    Parameters
    ----------
    dparams : DDMParams
        object holding all simulation parameters.

    Returns
    -------
    ims : list of 2D np.array
        stack of synthetic video frames.

    """
    psimul = ParticleSim2DBrownian(dparams)
    imgsynthez = ImageSynthesizer2D(psimul)
    Nframes = dparams.sim_Nt
    Npx = dparams.sim_img_Npx
    ims = np.zeros((Nframes,Npx,Npx),
                   dtype = np.float64)
    for ix in tqdm(range(Nframes)):
        img = imgsynthez.get_frame(ix)
        ims[ix,:,:] = img
    return ims



def simul1_save_simulation_result_file(fpn, videostack, ddmparams):
    """
    Write the result of a simulation to a file

    Parameters
    ----------
    fpn : str
        file pathname of the file to be written.
    videostack : list of 2D np.array (or 3D np.array)
        stack of synthetic video frames.
    ddmparams : DDMParams
        object holding all simulation parameters.

    Returns
    -------
    None.

    """
    print("Writing NPZ file with video and simulation parameters...")
    np.savez_compressed(fpn, 
                        videostack = videostack,
                        ddmparams = ddmparams)



def simul2_load_simulation_result_file(fpn):
    """
    Load simulation file and parameters

    Parameters
    ----------
    fpn : TYPE
        DESCRIPTION.

    Returns
    -------
    vid : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    """
    
    print('Loading file:', fpn)
    
    simulfile = np.load(fpn,
                        allow_pickle = True)
    vid = simulfile['videostack']
    params = simulfile['ddmparams'][()] # Ninja code to get object back.
                            # see: https://stackoverflow.com/a/8362451
                            #      https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
    simulfile.close()
    
    # set overdrive parameter (boost brightness)
    #TODO in parameter file?
    params.img_overdrive = 1.7
    
    return vid, params




def simul3_calculate_ISF(ims, params):
    """
    Calculate Image Structure Function from a simulated video

    Parameters
    ----------
    ims : list of 2D np.arrays (or a 3D np.array)
        Stack of video frames.
    params : DDMParams
        Object containing DDM Toolkit parameters.

    Returns
    -------
    ISF1 : ImageStructureFunction
        Data structure containing the Image Structure Function and relevant
        parameters.

    """

    
    ISE1 = ImageStructureEngine(params.ISE_Npx, params.ISE_Nbuf, 
                                engine_model = params.ISE_type)       

    for it in tqdm(range(params.Nframes)):
        ISE1.push(ims[it])
    
    ISF1 = ImageStructureFunction.fromImageStructureEngine(ISE1)
    
    # set real world units 
    ISF1.real_world(params.um_p_pix, params.s_p_frame)
    
    return ISF1
    



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

    dts = []
    D_fits = []
    allcloses = []
    for i,ISEmodel in enumerate(available_engine_models):
        print('Engine model #', ISEmodel)
        params.ISE_type = ISEmodel
        tstart = timer()
        ISF = simul3_calculate_ISF(ims, params)
        dt = timer() - tstart
        result = ISFanalysis_simple_brownian(ISF, params.initial_guess_D)
        dts.append(dt)
        D_fits.append(result.D_fit)
        if i==0:
            ISFref = ISF.ISF.copy()
            allcloses.append(True)
        else:
            allcloses.append(np.allclose(ISF.ISF, ISFref,
                                        rtol=5e-05, atol=1e-08))
        print('')
    return(available_engine_models, dts, D_fits, allcloses)