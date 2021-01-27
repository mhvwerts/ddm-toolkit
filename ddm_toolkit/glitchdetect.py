# -*- coding: utf-8 -*-

"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

glitchdetect.py:
A simple (but extensible) 'glitch detection' for raw input videos
based on DDM. The goal is to verify that the input video does not 
contain any 'glitches' (dropped frames, and the like).

It relies on calculating a stack of ISF at constant lag and then
analyzing that stack (SVD?)

In the present (basic) version, it returns the integrated intensity
of the one-frame lag differential image structure function

It requires to set a maximum number of processed ISFs
"""

import numpy as np
from .tifftools import TiffFile
from .ddm import ImageStructureEngine

class GlitchEngine:
    def __init__(self, ISF_Npx, Nframes, img0):
        self.ISF_Npx = ISF_Npx
        self.Nframes = Nframes
        # set up special ImageStructureEngine, and storage for results
        self.ISF_Nbuf = 1
        self.Ie = ImageStructureEngine(self.ISF_Npx, self.ISF_Nbuf) 
        self.tot = np.zeros(Nframes-1)
        self.img0 = img0
        self.i = 1
        
    def push(self, img1):
        self.img1 = img1
        self.Ie.reset()
        self.Ie.push(self.img0)
        self.Ie.push(self.img1)
        # =============================================================
        #     ISFstack[i-1] = Ie.ISFframe(1)
        # =============================================================
        self.tot[self.i-1] = np.sum(self.Ie.ISFframe(1)) 
        #TODO the total power(?) is a good parameter
        # but probably not infallible! 
        # BTW: what about Parseval??!!!!
        # a better glitch detection could rely on SVD or NIPALS
        # and measure the ratio of the weights of 1st and 2nd components
        self.img0, self.img1 = self.img1, self.img0
        self.i += 1



def glitchdetect_tiff(infp, ISF_Npx, Nframes = -1, Nskip = 0,
        img_offx = 0, img_offy = 0):
    """glitch detection on a TIFF file"""
    # =============================================================================
    # 
    # # resulting individual ISFs may be stored here
    # ISFstack = np.zeros((Ni-1, Ie.Npx, Ie.Npx),dtype=Ie.bufdtype)
    # 
    # =============================================================================
    #TODO
    # the following could probably be rewritten more elegantly
    # by changing 'ddm.py', in particular perhaps making a child class
    # that inherits from ImageStructureEngine, but instead of
    # 'accumulation', only returns the last ISF (tau) 
    # then we could also sample other time lags, because this present
    # video glitch tool only sample time lag = 1 frame
    #
    
    # open file
    print('glitchdetect: tiffile opening file')
    tif = TiffFile(infp)
    print('glitchdetect: tifffile creating iterator')
    tifit = iter(tif.pages)
    print('glitchdetect: tifffile getting file length')
    tiflen = len(tif.pages)
    print('TIFF length: {0:d} frames'.format(tiflen))
    print('glitchdetect: skipping frames (if any)')
    for i in range(Nskip):
         img_dummy = next(tifit)
    print('glitchdetect: skipped {0:d} frames '.format(Nskip))  
    if (Nframes < 0):
        Nframes = tiflen - Nskip - 1
        print('   processing full video')
    else:
        print('   processing', Nframes, 'frames')
   
    img0 = next(tifit).asarray()[img_offy:img_offy+ISF_Npx,
                              img_offx:img_offx+ISF_Npx]
    gld = GlitchEngine(ISF_Npx, Nframes, img0)

    for i in range(1, Nframes):
        img1 = next(tifit).asarray()[img_offy:img_offy+ISF_Npx,
                                  img_offx:img_offx+ISF_Npx]
        gld.push(img1)
    tif.close()
    return gld.tot


def glitchdetect_array(img, Nframes = -1):
    if (Nframes < 0):
        Nframes = img.shape[0] - 1
        print('   processing full video stack')
    else:
        print('   processing', Nframes,'frames')
    img0 = img[0]
    gld = GlitchEngine(img.shape[1], Nframes, img0)
    for i in range(1, Nframes):
        img1 = img[i]
        gld.push(img1)
    return gld.tot
    
