#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddm_numba_cuda.py: Numba-accelerated computations (CUDA GPU)

@author: werts
"""

import numpy as np
import numba.cuda as cuda


# This CUDA-accelerated ImageStructureEngine is a class of its own, in a module
# of its own.
#
# I give you 'ImageStructureEngine7'

class ImageStructureEngine7:
    """A CUDA-accelerated ImageStructureEngine (Numba)
 
    """
    def __init__(self, Npx, Nbuf):
        # get parameters: image size (Npx times Npx) and ISF depth (Nbuf)
        self.Npx = Npx
        self.Nbuf = Nbuf

        # index of last written circular buffer memory slot
        self.ix_in = 0 
        # number of frames pushed to buffer
        self.totalframes = 0
        # number of frames contributing to ISF (for normalization)
        self.ISFcount = 0
        
        # set frame/pixel scaling
        self.tauf = np.arange(self.Nbuf+1) * 1.0
        self.ux = np.fft.fftshift(np.fft.fftfreq(self.Npx))
        self.uy = np.fft.fftshift(np.fft.fftfreq(self.Npx))

        # Buffers on the GPU
        # 
        # This uses the numba.cuda GPU memory management API
        # See: https://numba.pydata.org/numba-doc/latest/cuda/memory.html
        #

        # Circular buffer for FFT images
        self.BUFreal = cuda.device_array((self.Nbuf, self.Npx, self.Npx),
                                         dtype=np.float32)
        self.BUFimag = cuda.device_array((self.Nbuf, self.Npx, self.Npx),
                                         dtype=np.float32)
        # incoming frame buffer
        self.img_in_fft_real = cuda.device_array((self.Npx, self.Npx),
                                                 dtype=np.float32)
        self.img_in_fft_imag = cuda.device_array((self.Npx, self.Npx),
                                                 dtype=np.float32)
        # ISF: image structure function array (cumulative)
        self._ISFaccum = cuda.device_array((self.Nbuf, self.Npx, self.Npx),
                                      dtype=np.float32)

        # CUDA-related settings:
        #TO BE OPTIMIZED....
        # see: https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
        self.BL = 16 # block length (height, width)
        assert self.Npx%self.BL == 0,\
          "ImageStructureEngine7: Npx should be multiple of BL (adapt BL or Npx)"
        self.BPG = (self.Npx//self.BL,self.Npx//self.BL) # this just takes the entire image
        self.TPB = (self.BL,self.BL) # 256 = multiple of 32
        
        # the following properties have no use in this engine
        # they are (for now) need for compatibility with first-generation code
        self.Npick = 1
        self.Navg = 1
        self.Ndrop = 0
        self.apodwindow = None


    def push(self, img_in):
        """Push frame to Image Structure Engine
        
        This outside function handles the circular (FIFO) buffer.
        It then calls the image structure calculation function
        
        """
       
        # TODO: incoming frame: apply apodization (Blackman-Harris window etc.)

        # process incoming frame: FFT2D and transfer to GPU
        # in future (GPU async) version: first update ISF buffer
        #  and only at the end do FFT (such that CPU FFT while GPU calculating)
        img_in_fft = np.fft.fft2(img_in) # fft may be sped up? fft2 not in numba.
    
        fft_real = img_in_fft.real.copy() # necessary step
        fft_imag = img_in_fft.imag.copy()

        self.img_in_fft_real[:,:] = fft_real
        self.img_in_fft_imag[:,:] = fft_imag

        # only when all (circular) buffer positions have been filled
        if (self.totalframes >= self.Nbuf):
            # img_in is frame "t + dt"
            # frame in buffer is frame "t"
            _gpu_ISF_push_frame_fft[self.BPG,self.TPB](
                                 self.img_in_fft_real, self.img_in_fft_imag,
                                 self.BUFreal, self.BUFimag, self._ISFaccum,
                                 self.ix_in, self.Nbuf)
            self.ISFcount += 1

        # update index for writing incoming frame (FFT2) to circular buffer
        if (self.ix_in <= 0):
            self.ix_in = self.Nbuf
        self.ix_in -= 1
        
        # write fft  of incoming image to buffer slot ix_in
        # GPU: this should be done using buffers in GPU memory
        # TODO: check if the following works entirely ON THE GPU or if it passes via the host... 
        #  perhaps we should use a specific GPU-based memcopy?
        #  on the other hand: it seems to work
        self.BUFreal[self.ix_in, :, :] = self.img_in_fft_real[:,:]
        self.BUFimag[self.ix_in, :, :] = self.img_in_fft_imag[:,:]
        
        # frame counter
        self.totalframes += 1

                
    def ISF(self):
        """return a new matrix with the full, correctly oriented
        ('fftshift'-ed) ISF[t,y,x]"""
        assert self.ISFcount > 0, 'No ISF is available (not enough '\
                                  'image frames to fill buffer?)'
            
        ISFarray = np.zeros((self.Nbuf+1,self.Npx,self.Npx)) 
        # in ImageStructureFunction ISFarray, the array is offset by 1
        # this is because ISF[0] contains delta t = 0
        for ix, dISF in enumerate(self._ISFaccum):
            ISFarray[ix+1,:,:] = np.fft.fftshift(dISF)/self.ISFcount
    
        return ISFarray



# The GPU kernel is kept as a separate function outside of the class
# all relevant data pointers are passed as arguments, so no need to use 'self.'

@cuda.jit
def _gpu_ISF_push_frame_fft(gpu_in_fft_real, gpu_in_fft_imag, 
                        gpu_BUFreal, gpu_BUFimag, gpu_ISF,
                        ix_in, Nbuf):
    """process an incoming frame for the image structure function
    
    CUDA kernel.
    
    """

    xx,yy = cuda.grid(2)

    for n in range(Nbuf):
        ixc = (n + ix_in) % Nbuf
        dISF = (gpu_in_fft_real[xx,yy] - gpu_BUFreal[ixc,xx,yy])**2 \
             + (gpu_in_fft_imag[xx,yy] - gpu_BUFimag[ixc,xx,yy])**2
        # add to total isf
        gpu_ISF[n,xx,yy] += dISF

