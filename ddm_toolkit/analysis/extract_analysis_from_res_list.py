import numpy as np

def extract_analysis_from_res_list(res_list):
    '''
    

    Parameters
    ----------
    res_list : DATA STRUCT
        Data structure provided after ISF fitting procedure.

    Returns
    -------
    D : ARRAY (Numpy)
        Diffusion coefficient.
    A_q : ARRAY (Numpy)
        Sample scattering intensity modulaed by the OTF of the microscope.
    B_q : ARRAY (Numpy)
        Detection system noise.
    q : ARRAY (Numpy)
        Spatial frequency vectoe.

    '''
    
    D = np.asarray([res_list[i].D_fit for i in range(0,len(res_list)) ])
    A_q = np.asarray([res_list[i].A_q for i in range(0,len(res_list)) ])
    B_q = np.asarray([res_list[i].B_q for i in range(0,len(res_list)) ])
    q = np.asarray(res_list[0].q)
    
    
    return D, A_q, B_q, q
