#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from sys import argv
from configparser import ConfigParser

import numpy as np

from ddm_toolkit.tifftools import TiffFile, tifffile_version
from ddm_toolkit.videofig import videofig
from ddm_toolkit import tqdm


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
# get video info

print('file being processed: ', infp)


tif = TiffFile(infp)

tiflen = len(tif.pages) - 1 # remove last frame (sometime causes problems)
print('TIFF length: {0:d} frames'.format(tiflen))
# get first image in file in order to determine dimension
img = tif.pages[0].asarray()
print('frame shape: {0:d} x {1:d}'.format(*img.shape))
# enable full image processing
if ROI_size < 0:
    ROI_x2 = img.shape[1]
    ROI_y2 = img.shape[0]

frm_prevend = frm_start + frm_Npreview
if frm_prevend > tiflen:
    frm_prevend = tiflen
frm_Npreview = frm_prevend - frm_start

vidshape = (frm_Npreview,img.shape[0],img.shape[1])
vid = np.zeros(vidshape)

print('loading preview frames into memory')
for i in tqdm(range(frm_Npreview)):
    img = tif.pages[frm_start + i].asarray()
    # copy all pixels divided by ROIcontrast (low intensity)
    vid[i,:,:] = img[:,:] / ROIcontrast
    # only copy ROI zone at full intensity
    vid[i,ROI_y1:ROI_y2,ROI_x1:ROI_x2] = img[ROI_y1:ROI_y2,ROI_x1:ROI_x2]

tif.close()

#%% display video

vmx = vid.max() / vid_overdrive # avoid autoscale of colormap
vf_redraw_init=False
vf_redraw_img=None
def vf_redraw(fri, ax):
    global vid
    global vmx
    global vf_redraw_init
    global vf_redraw_img
    if not vf_redraw_init:
        vf_redraw_img=ax.imshow(vid[fri], 
                                vmin = 0.0, vmax = vmx, 
                                origin = 'lower', animated = True)
        vf_redraw_init=True
    else:
        vf_redraw_img.set_array(vid[fri])

print("[ENTER]: toggle pause/play")
print("[LEFT]/[RIGHT]: scroll frames")
print("[MOUSE]: manipulate time bar")

videofig(frm_Npreview, vf_redraw, play_fps=10)



