#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:09:48 2022

@author: werts
"""
from ddm_toolkit import tqdm

from ddm_toolkit.ddm import ImageStructureEngine
from ddm_toolkit.ddm import ImageStructureFunction
from ddm_toolkit.ddm import best_available_engine_model

def calculate_ISF(framestrm, Nbuf = 200, Nframes = 1000, ISE_type = -1):
    """Calculate Image Structure Function from incoming video frames

    Parameters
    ----------
    framestrm : ddm_toolkit.framestreamers.FrameStreamer
        Incoming video stream (via FrameStreamer instance).
    Nbuf : int, optional
        Size of DDM buffer (number of lag points). The default is 200.
    Nframes : int, optional
        Number of frames to be analyzed. The default is 1000.
    ISE_type : int, optional
        ImageStructureEngine type. The default is -1 (Select fastest engine)

    Returns
    -------
    ISF : ddm_toolkit.ddm.ImageStructureFunction
        ISF of incoming video stream.
    """
    framestrm.rewind()

    if ISE_type < 0:
        ISE_type = best_available_engine_model
    print('ImageStructureEngine type n° : ',ISE_type)
    #TO DO: apodization (as part of FrameStreamer)
    ISE = ImageStructureEngine(framestrm.ROI_iw, Nbuf,
                            engine_model = ISE_type)


    # loop over selected images in file
    for i in tqdm(range(Nframes)):
        img = framestrm.next_frame(return_ROI = True)
        ISE.push(img)

    ISF = ImageStructureFunction.fromImageStructureEngine(ISE)
    return ISF
