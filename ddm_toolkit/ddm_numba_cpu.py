#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddm_numba_cpu.py: Numba-accelerated computations (CPU)

@author: werts
"""

import numba
import numpy as np
from matplotlib import pyplot as plt

@numba.jit(nopython=True)
def numba_ISF_push_frame_fft_v1(img_in_fft_real, img_in_fft_imag, _ISF,
                        ix_in, BUFreal, BUFimag, Nbuf):
    """process an incoming frame for the image structure function

    Numba nJIT single-threaded version

    """
    for n in range(Nbuf):
        ixc = (n + ix_in) % Nbuf
        dISF = (img_in_fft_real - BUFreal[ixc,:,:])**2 \
              + (img_in_fft_imag - BUFimag[ixc,:,:])**2
        # add to total isf
        _ISF[n,:,:] += dISF



@numba.jit(nopython=True, parallel=True)
def numba_ISF_push_frame_fft_v2(img_in_fft_real, img_in_fft_imag, _ISF,
                        ix_in, BUFreal, BUFimag, Nbuf):
    """process an incoming frame for the image structure function

    Numba nJIT parallel version. Multithreading if available.

    """

    for n in numba.prange(Nbuf):
        ixc = (n + ix_in) % Nbuf
        dISF = (img_in_fft_real - BUFreal[ixc,:,:])**2 \
              + (img_in_fft_imag - BUFimag[ixc,:,:])**2
        # add to total isf
        _ISF[n,:,:] += dISF




@numba.jit(nopython=True, parallel=True)
def numba_ISF_push_frame_fft_v2_corr_LB(img_in_fft_real, img_in_fft_imag, _ISF,
                        ix_in, BUFreal, BUFimag, Nbuf, v_pix_s1, phi_0, ux, uy,tau):
    """process an incoming frame for the image structure function

    modified by Lancelot Barthe

    Numba nJIT parallel version. Multithreading if available.

    """


    """TODO VERIFY COMPUTATION OF v_dot_u and if the multiplication is an elemen wise one """
    
    v_dot_u = np.zeros((ux.size,uy.size))

    for i in range(ux.size):
        for j in range(uy.size):
            v_dot_u[i,j] = v_pix_s1*np.cos(phi_0)*ux[i] + v_pix_s1*np.sin(phi_0)*uy[j]


    for n in numba.prange(Nbuf):
        ixc = (n + ix_in) % Nbuf

        img_in_fft_real_temp = img_in_fft_real*np.cos(-v_dot_u*tau[n]) - img_in_fft_imag*np.sin(-v_dot_u*tau[n])
        img_in_fft_imag_temp = img_in_fft_imag*np.cos(-v_dot_u*tau[n]) + img_in_fft_real*np.sin(-v_dot_u*tau[n])

        img_in_fft_real = img_in_fft_real_temp
        img_in_fft_imag = img_in_fft_imag_temp

        dISF = (img_in_fft_real - BUFreal[ixc,:,:])**2 \
              + (img_in_fft_imag - BUFimag[ixc,:,:])**2
        # add to total isf
        _ISF[n,:,:] += dISF
