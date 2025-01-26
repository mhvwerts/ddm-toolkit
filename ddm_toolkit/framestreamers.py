#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from io import BytesIO
import numpy as np
import PIL.Image
import av
from .tifftools import TiffFile

class FrameStreamerBasic:
    def __init__(self, viewport, Nskipstart = 0):
        self.viewport = viewport
        self.vmin = -5.0
        self.vmax = +5.0
        self.abuf = np.zeros(self.viewport, dtype = np.uint8)
        self.framedata = np.zeros(self.viewport)
        self.intscale = 256
        self.imgmode = 'L' # this goes with np.uint8 monochrome pixel values
        self.ROI_active = False
        self.ROI_ix = 0
        self.ROI_iy = 0
        self.ROI_iw = 32  # width/length of (square) ROI, must be a value available in the UI drop-down
        self.playing = False
        self.frameix = 0
        self.Nskipstart = Nskipstart # skip the first Nskipstart frames from the start ('rewind + fastforward')
        self.Nframes = 1000 # virtual number of frames
        self.random_access = True # does the image data source enable random access (e.g. memory, TIFF file)
                                  # or is sequential only (e.g. AVI file)
        # Fast-forward to frame# Nskipstart
        self.rewind()
        
    def imgbytes(self, fmt='png'):
        """Render a 2D array as an image file (PNG) in a byte string
        
        (for passing to Image Widget)
        """
        # apply contrast/brightness (via vmin, vmax) and generate 256 grayscale 
        # image 'abuf'
        if self.vmin == self.vmax: # avoid division by zero
            self.abuf[:,:] = self.intscale - 1
        else:
            self.abuf[:,:] = (self.framedata[:,:] - self.vmin)\
                                /(self.vmax - self.vmin) * self.intscale
        # limiters
        self.abuf[self.framedata<=self.vmin] = 0
        self.abuf[self.framedata>=self.vmax] = 255
        # diminish intensity outside of ROI
        if self.ROI_active:
            #TODO check that ROI_iy etc. are within acceptable range?
            # ... not really necessary, yet
            self.abuf[:self.ROI_iy,:] //= 2
            self.abuf[(self.ROI_iy+self.ROI_iw):,:] //= 2
            self.abuf[self.ROI_iy:(self.ROI_iy+self.ROI_iw),
                      :self.ROI_ix] //= 2
            self.abuf[self.ROI_iy:(self.ROI_iy+self.ROI_iw),
                      (self.ROI_ix+self.ROI_iw):] //= 2
        with BytesIO() as f:
            img = PIL.Image.fromarray(self.abuf, mode=self.imgmode)
            # we could implement a palette later on. 
            # for now, always use 256 gray scale (np.uint8)
            img.save(f, fmt)
            imgbytes = f.getvalue()
        return imgbytes
    
    def next_frame(self, return_ROI=False):
        if self.frameix >= self.Nframes:
            # self.frameix = 0 # do not cycle around anymore, simulate finite source
            outdata = None # output None if last frame reached
        else:
            self.framedata = self._get_next_framedata(self.frameix)
            self.frameix += 1
            if return_ROI:
                outdata = self.framedata[self.ROI_iy:self.ROI_iy+self.ROI_iw,
                                         self.ROI_ix:self.ROI_ix+self.ROI_iw]    
            else:
                outdata = self.framedata
        return outdata
    
    def _get_next_framedata(self, ix):
        imagedata = np.random.normal(size=self.viewport)
        return imagedata
        
    def rewind(self):
        """
        Return to start of video file
        
        (Then, automatically forwards Nskipstarts frames)

        Returns
        -------
        None.

        """
        if self.Nskipstart > 0:
            self.frameix = self.Nskipstart
        else:
            self.frameix = 0
     


class FrameStreamer_ndarray(FrameStreamerBasic):
    """Stream frames from a numpy array
    
    The numpy array is organized as follows
        arr[ti, xi, yi]
    such that
        arr[ti, :, :]
    is a single monochrome frame
    """
    def __init__(self, arr, **kwargs):
        self.arr = arr
        self.frameshape = arr.shape[1:]
        self.globalmax = arr.max()
        self.globalmin = arr.min()
        super().__init__(self.frameshape, **kwargs)
        self.Nframes = arr.shape[0]
        self.random_access = True
        self.vmin = self.globalmin
        self.vmax = self.globalmax  
        
    def _get_next_framedata(self, ti):
        imagedata = self.arr[ti, :, :]
        return imagedata



class FrameStreamerTIFF(FrameStreamerBasic):
    def __init__(self, fname, **kwargs):
        self.tf = TiffFile(fname)
        page = self.tf.pages[0]
        frm = page.asarray()
        self.frameshape = page.shape
        self.globalmax = frm.max()
        self.globalmin = frm.min()
        super().__init__(self.frameshape, **kwargs)
        self.Nframes = len(self.tf.pages)
        self.random_access = True # TiffFile provides random access through the TIFF pages
        self.vmin = self.globalmin
        self.vmax = self.globalmax
                    
    def _get_next_framedata(self, ix):
        imagedata = self.tf.pages[self.frameix].asarray()
        return imagedata




class FrameStreamerAVI(FrameStreamerBasic):
    def __init__(self, fname, Nframes, colorchannel=0, **kwargs):
        self.colorchannel = colorchannel # select one color channel from RGB MJPEG AVI (to do: monochrome conversion)
        self.container = av.open(fname)
        self.vidstream = self.container.decode(video=0)
        self.viditer = iter(self.vidstream)
        arr = self._get_next_framedata(0) # sacrifice first image to get shape etc.
        self.frameshape = arr.shape
        self.globalmax = arr.max()
        self.globalmin = arr.min()
        super().__init__(self.frameshape, **kwargs)
        self.Nframes = Nframes # set by hand for now, 
                               # find a way to get a good estimate of number of frames in video.
                               # setting this simply to a high number (10000) will work in most use cases
        self.random_access = False # AVI in PyAV is not reliably random access.
        self.vmin = self.globalmin
        self.vmax = self.globalmax
                
    def _get_next_framedata(self, ix):
        """
        ix is not used here!
        """
        try:
            frame = next(self.viditer)             
        except StopIteration:
            frame = None
        if frame is not None:
            img = frame.to_image()
            fullarr = np.asarray(img)
            imagedata = fullarr[: , :, self.colorchannel] # take only one color channel
        else:
            imagedata = None # last image detected
        return imagedata

    def rewind(self):
        """
        Return to start of video file

        Returns
        -------
        None.

        """
        self.frameix = 0
        self.container.seek(0)
        
        # then forward Nskipstart frames
        if self.Nskipstart > 0:
            print('AVI: Fast forwarding...', self.Nskipstart, 'frames')
            for i in range(self.Nskipstart):
                self.next_frame()
        
        
        
        
        
