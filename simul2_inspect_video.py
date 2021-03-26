#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# The DDM Team, 2020-2021
#
# diffusion coefficient in => diffusion coefficient out
#
# STEP 2: Play synthetic video (videofig player)
#


import numpy as np

from ddm_toolkit.videofig import videofig 
from ddm_toolkit import sim_params


# Get parameters
sim = sim_params()

# overdrive parameter (boost brightness)
#TODO in parameter file?
img_overdrive = 1.7

# LOAD and VISUALIZE video stack
im=np.load(sim.vidfpn)
img=im['img']
im.close()
if (sim.Nview < 0):
    Ni = img.shape[0]
else:
    Ni = sim.Nview
    
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


