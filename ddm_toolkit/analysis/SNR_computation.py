#
import numpy as np

#
def SNR_computation(q, avg_Aq, avg_Bq):
    '''
    

    Parameters
    ----------
    q : ARRAY (Numpy)
        Spatal frequency vector.
    avg_Aq : ARRAY (Numpy)
        Averaged results of several fitting procedure for A(q).
    avg_Bq : ARRAY (Numpy)
        Averaged results of seeral fitting procedure for B(q).

    Returns
    -------
    SNR : ARRAY (Numpy)
        Signal to Noise Ratio.

    '''


    delta_q = q[1]

    P_tot_Aq = np.sum(avg_Aq)*delta_q
    P_tot_Bq = np.sum(avg_Bq)*delta_q

    SNR = P_tot_Aq / P_tot_Bq

    return SNR
