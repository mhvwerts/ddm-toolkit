from timeit import default_timer as timer
import numpy as np
from ddm_toolkit.ddm import available_engine_models
from ddm_toolkit.analysis import ISFanalysis_simple_brownian
from .simul3_calculate_ISF import simul3_calculate_ISF

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
        print('D_fit = ',result.D_fit)
        if i==0:
            ISFref = ISF.ISF.copy()
            allcloses.append(True)
        else:
            allcloses.append(np.allclose(ISF.ISF, ISFref,
                                        rtol=5e-05, atol=1e-08))
        print('')
    return(available_engine_models, dts, D_fits, allcloses)


