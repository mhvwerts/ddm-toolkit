
#
def a_priori_noise_detector_model(q, γ_0, qc ):
    '''
    Model :
        0° Modulus of wave vector :
            q = sqrt(qx^2 + qy^2)

        1° Spectral power density of white noise :
            γ_WN(q) = γ_0/2

        2° Spectral power density of pink noise :
            γ_PN(q) = α qc/q

        3° Total spectral power density :
            B(q)= γ_0/2 (1 + qc/q)

    Parameters
    ----------
    q : ARRAY (Numpy)
        Spatial pulsation.
    γ_0 : FLOAT
        Power Spectral Density constant.
    qc : FLOAT
        Spatial cutting pulsation

    Returns
    -------
    B_qq : ARRAY (Numpy)
        A-priori model of the Noise term of DDM eq..

    '''
    B_qq = (γ_0/2)*(1 + (qc/q))

    return B_qq
