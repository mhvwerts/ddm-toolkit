import numpy as np

def IQR_outliers_removal(res_list):
    '''
    

    Parameters
    ----------
    res_list : DATA STRUCT
        Results provided by ISF fitting procedure.

    Returns
    -------
    D : ARRAY (Numpy)
        Diffusion coefficient.
    D_no_outliers : ARRAY (Numpy)
        Diffusion coefficient after removing outlies.
    A_q : ARRAY (Numpy)
        Scattering intensity modulated with microscope OTF.
    A_q_no_outliers : ARRAY (Numpy)
    cattering intensity modulated with microscope OTF after removing outliers.
    B_q : ARRAY (Numpy)
        Noise of the detection system.
    B_q_no_outliers : ARRAY (Numpy)
        Noise of the detection system after outliers removal.
    i_outliers : ARRAY (Numpy)
        Indices of the detected outliers in the data set.
    q : ARRAY (Numpy)
        Spatial frequency vector.

    '''
    
    D = np.asarray([res_list[i].D_fit for i in range(0,len(res_list)) ])
    A_q = [res_list[i].A_q for i in range(0,len(res_list)) ]
    B_q = [res_list[i].B_q for i in range(0,len(res_list)) ]
    q = res_list[0].q
    
    # Define IQR
    Q1 = np.percentile(D, 25,interpolation = 'midpoint')
    Q3 = np.percentile(D, 75,interpolation = 'midpoint')
    IQR = Q3 - Q1
    
    # Determine lower and upper bound for IQR outliers removal
    upper = D >= (Q3+1.5*IQR) # Above Upper bound
    lower = D <= (Q1-1.5*IQR) # Below Lower bound
    
    # Generate mask using lower and upper bound
    mask = upper | lower
    
    # Determine indices where outliers are in array
    i_outliers = np.where(mask)[0]
    print(i_outliers)
    
    # Complement mask
    mask = np.logical_not(mask)
    
    # Apply mask to diffusion vector
    D_no_outliers = D[ mask ,...]
    
    # Apply mask to other array
    A_q_no_outliers = A_q
    B_q_no_outliers = B_q
    i_outliers_temp = i_outliers
    
    i = len(i_outliers) - 1
    while len(i_outliers_temp)>=1:
        A_q_no_outliers.pop(i_outliers_temp[i])
        B_q_no_outliers.pop(i_outliers_temp[i])
        i_outliers_temp =  np.delete(i_outliers_temp, i)
        i -=1
    
    return D, D_no_outliers, A_q, A_q_no_outliers, B_q, B_q_no_outliers,i_outliers,q
