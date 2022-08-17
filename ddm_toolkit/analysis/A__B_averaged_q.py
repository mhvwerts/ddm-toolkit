#
import numpy as np

#
def A__B_averaged_q(A, B):
    '''
    

    Parameters
    ----------
    A : ARRAY (Numpy)
        Sample scatterig intensity modulated by the OTF of the microscope.
    B : ARRAY (Numpy)
        Acquisition system Noise.

    Returns
    -------
    avg_Bq : FLOAT
        Averaged version of B.
    avg_Aq : FLOAT
        Averaged version of A.
    i_true : INT
        Zone of q where data are available.

    '''
    # Average every A(q) and B(q) and determine zone of interest
    avg_Bq = np.average(B,axis=0)
    avg_Aq = np.average(A,axis=0)
    i_true = np.where(avg_Bq!=0)

    return avg_Bq, avg_Aq, i_true
