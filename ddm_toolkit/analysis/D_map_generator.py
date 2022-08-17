import numpy as np

def D_map_generator(D, map_size):
    '''
    

    Parameters
    ----------
    D : ARRAY (Numpy)
        Diffusion coefficient fitted.
    map_size : INT
        Nuber of pixel defining a square map.

    Returns
    -------
    D_map : ARRAY (Numpy)
        Reshape vector into matrice.

    '''
        
    D_map =  np.reshape(D,(map_size,map_size))
    return D_map
