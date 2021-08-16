#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddm_numba_cpu.py: Numba-accelerated computations (CPU)

@author: werts
"""

import numba

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