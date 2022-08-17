#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit import tqdm


class check_videoROI_result:
    """Results of check_videoROI run

    Attributes
    ----------
    imgItot : numpy.ndarray (1D vector)
        Integrated intensity for each analyzed frame.
    E_dimg : numpy.ndarray (1D vector)
        Energy of differences between frames (lag: 1 frame).
    E_dimg2 : numpy.ndarray (1D vector)
        Energy of differences between frames (lag: 2 frames).
    Nframes : int
        Number of frames analyzed.
    img_shape : tuple
        Shape of frames.
    """
    
    def __init__(self, imgItot, E_dimg, E_dimg2, Nframes, shp):

        self.imgItot = imgItot
        self.E_dimg = E_dimg
        self.E_dimg2 = E_dimg2
        self.Nframes = Nframes
        self.img_shape = shp
        
    def plot_intensity(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.imgItot)
        plt.xlabel('frame#')
        plt.ylabel('total intensity')
        plt.title('total intensity evolution')
        plt.ylim(ymin = 0.0,
                 ymax = self.imgItot.max()*1.2)
        #print('ROI shape=',self.img_shape)
        plt.figure(2)
        plt.clf()
        plt.hist(self.imgItot, 
                 bins=max(15, self.Nframes//40),
        #         range=(0.0, imgItot.max())
                 )
        plt.xlabel('intensity')
        plt.ylabel('#frames')
        plt.title('intensity histogram')
        
    def plot_diff_energy(self):
        plt.figure(3)
        plt.clf()
        plt.plot(self.E_dimg2, label = 'lag=2')
        plt.plot(self.E_dimg, label = 'lag=1')
        plt.ylim(ymin = 0.0,
                 ymax = self.E_dimg.max()*1.2)
        plt.title('Energy of frame difference images')
        plt.xlabel('frame pair#')
        plt.ylabel('total energy of difference image')
        plt.legend()
        
        plt.figure(4)
        plt.clf()
        plt.hist(self.E_dimg2,
                 bins=max(20, self.Nframes//40)
                 )
        plt.hist(self.E_dimg,
                 bins=max(20, self.Nframes//40)
                 )
        
        plt.title('Energy of frame difference images')
        plt.xlabel('energy (difference image)')
        plt.ylabel('#frame pairs')
        
        
        

def check_videoROI(framestrm, Nframes = 1000):
    """
    Check videos via frame intensity and frame differences

    Parameters
    ----------
    framestrm : ddm_toolkit.framestreamers.FrameStreamer object
        FrameStreamer object streaming frames from a data source.
    Nframes : int, optional
        Number of frames to be analyzed. The default is 1000.

    Returns
    -------
    check_videoROI_result object
        Object containing analysis results and methods for displaying them.

    """

    #TODO Nskip

    
    framestrm.rewind() # skip to beginning of stream

    imgItot = np.zeros(Nframes)
    E_dimg = np.zeros(Nframes-1)
    E_dimg2 = np.zeros(Nframes-2)
    previmg = 0.0 # just initialize this as a scalar value
    previmg2 = 0.0 

    for i in tqdm(range(Nframes)):
        img = framestrm.next_frame(return_ROI = True)

        # The total intensity is simply the sum over all
        # pixels in an image frame
        imgItot[i] = np.sum(img)
 
        # Here we calculate the difference between two
        # subsequent frames (frame pair #0 = frame#0 and frame#1)
        # and then take the 'energy' (sum over squares)
        # This helps in detecting any glitches in the video (skipped
        # frames etc.)
        if i>0:
            dimg = (img - previmg)
            E_dimg[i-1] = np.sum(dimg**2)
        if i> 1:
            dimg2 = (img - previmg2)
            E_dimg2[i-2] = np.sum(dimg2**2)
        previmg2 = previmg
        previmg = img

    return check_videoROI_result(imgItot, E_dimg, E_dimg2, Nframes, img.shape)