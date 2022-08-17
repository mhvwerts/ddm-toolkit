import numpy as np


def D_ext(res_list):
    '''
    

    Parameters
    ----------
    res_list : DATA STRUCT
        Data structure provided after fitting proceure of ISFs.

    Returns
    -------
    D : ARRAY (Numpy)
        Diffusion coefficient fitted for each ISFs.

    '''
    
    D =  np.zeros(len(res_list))
    
    for i in range(0,len(res_list)):
      D[i] = res_list[i].D_fit
    
    return D
