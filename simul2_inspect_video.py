#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# by Greshma Babu and Martinus Werts, 2020
#
# diffusion coefficient in => diffusion coefficient out
#
# STEP 2: Play synthetic video (videofig player)
#

from sys import argv
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit.videofig import videofig 


# Read parameter file, default to "simul0_default_params.txt"
# if nothing
params = ConfigParser()
argc = len(argv)
if argc == 1:
    parfn = "simul0_default_params.txt"
elif argc == 2:
    parfn = argv[1]
else:
    raise Exception('invalid number of arguments')
params.read(parfn)
videof = params['imgsynth']['img_file']
Nframes = int(params['animation']['Nframes'])

# overdrive parameter (boost brightness)
#TODO in parameter file?
img_overdrive = 1.8

# LOAD and VISUALIZE video stack
im=np.load(videof)
img=im['img']
im.close()
if (Nframes < 0):
    Ni = img.shape[0]
else:
    Ni = Nframes
    
vmx = img.max() / img_overdrive # avoid autoscale of colormap

vf_redraw_init=False
vf_redraw_img=None
def vf_redraw(fri, ax):
    global img
    global vmx
    global vf_redraw_init
    global vf_redraw_img
    if not vf_redraw_init:
        vf_redraw_img=ax.imshow(img[fri], 
                                vmin = 0.0, vmax = vmx, 
                                origin = 'lower', animated = True)
        vf_redraw_init=True
    else:
        vf_redraw_img.set_array(img[fri])

print("[ENTER]: toggle pause/play")
print("[LEFT]/[RIGHT]: scroll frames")
print("[MOUSE]: manipulate time bar")

videofig(Ni, vf_redraw, play_fps=10)


