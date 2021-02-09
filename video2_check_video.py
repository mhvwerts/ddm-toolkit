#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: werts

This is a utility script for quantitatively checking the quality
of a recorded video. It plots some basic metrics like the total intensity
for each frame. 

It also uses a basic form of glitch detection (frame skipping) by measuring
the total energy (sum of intensity**2 over all pixels) of the difference image
between subsequent frames.

Energy (signal processing): 
    https://en.wikipedia.org/wiki/Energy_(signal_processing)


For now, it just plots these measurements for visual inspection. It is up to
the operator to decide if there is a problem in the file

#TODO: see if we can define some quantitative numerical measure to automate
detection of problems (outlier detection)

This script uses 'tifffile' by C. Gohlke
More documentation on this library:
    https://github.com/cgohlke/tifffile
(especially in the "Examples" section of 'README.rst')




python video2_check_video.py <filename.txt>
    this will use a configuration file, which enables us
    to set a ROI window, a start and stop frame etc.


"""

from sys import argv
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit.tifftools import TiffFile, tifffile_version
from ddm_toolkit.tqdm import tqdm

print('tifffile version: ', tifffile_version)


#%% 
# ==============================
# get PROCESSING PARAMETERS
# ==============================
# Read parameter file, default to "video0_test_params.txt"
# if nothing (easier with Spyder)

argc = len(argv)
if argc == 1:
    argfn = "video0_test_params.txt"
elif argc == 2:
    argfn = argv[1]
else:
    raise Exception('invalid number of arguments')
 
params = ConfigParser(interpolation=None)
params.read(argfn)

infp = params['videofile']['pathname']
frm_start = int(params['videofile']['frm_start'])
frm_end = int(params['videofile']['frm_end'])
ROI_x1 = int(params['videofile']['ROI_x'])
ROI_y1 = int(params['videofile']['ROI_y'])
ROI_size = int(params['videofile']['ROI_size'])
frm_Npreview = int(params['videofile']['frm_Npreview'])
vid_overdrive = float(params['videofile']['display_overdrive'])
ROIcontrast = float(params['videofile']['display_ROIcontrast'])
ROI_x2 = ROI_x1 + ROI_size
ROI_y2 = ROI_y1 + ROI_size


#%% load video and apply ROI



print('file being processed: ', infp)
tif = TiffFile(infp)

tiflen = len(tif.pages) - 1 # remove last frame (sometime causes problems)
print('TIFF length: {0:d} frames'.format(tiflen))

# first image to get dimensions
imgfull = tif.pages[0].asarray()
# enable full image processing
if ROI_size < 0:
    ROI_x2 = imgfull.shape[1]
    ROI_y2 = imgfull.shape[0]

print('ROI: x={0:d}...{1:d}  y={2:d}...{3:d}'.format(ROI_x1,
                                                     ROI_x2,
                                                     ROI_y1,
                                                     ROI_y2))
if (frm_end > tiflen) or (frm_end < 1):
    frm_end = tiflen

print('selected frames: start={0:d}  end={1:d}'.format(frm_start,
                                                       frm_end))
Nframes = frm_end - frm_start
imgItot = np.zeros(Nframes)
E_dimg = np.zeros(Nframes-1)
previmg = 0.0 # just initialize this as a scalar value
# loop over selected images in file
# tqdm just makes a progress bar (useful for large files!)
for i in tqdm(range(Nframes)):
    imgfull = tif.pages[i + frm_start].asarray()
    # get ROI only
    img = imgfull[ROI_y1:ROI_y2,ROI_x1:ROI_x2]
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
        previmg = img
    else:
        previmg = img
        
tif.close()

plt.figure(1)
plt.clf()
plt.plot(imgItot)
plt.xlabel('frame#')
plt.ylabel('total intensity')
plt.title('total intensity evolution')
plt.ylim(ymin = 0.0,
         ymax = imgItot.max()*1.2)


plt.figure(2)
plt.clf()
plt.hist(imgItot, 
         bins=max(15, tiflen//40),
#         range=(0.0, imgItot.max())
         )
plt.xlabel('intensity')
plt.ylabel('#frames')
plt.title('intensity histogram')

plt.figure(3)
plt.clf()
plt.plot(E_dimg)
plt.ylim(ymin = 0.0,
         ymax = E_dimg.max()*1.2)
plt.title('Energy of frame difference images')
plt.xlabel('frame pair#')
plt.ylabel('total energy of difference image')

plt.figure(4)
plt.clf()
plt.hist(E_dimg,
         bins=max(20, tiflen//40)
         )

plt.title('Energy of frame difference images')
plt.xlabel('energy (difference image)')
plt.ylabel('#frame pairs')

print('Close all graph windows to end script...')
plt.show()



