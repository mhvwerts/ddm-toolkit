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
# STEP 4: Inspect ImageStructureFunction (videofig player)
#

from ddm_toolkit import ImageStructureFunction
from ddm_toolkit import sim_params

from ddm_toolkit.videofig import videofig 


#
# SIMULATION/ANALYSIS PARAMETERS
#
sim = sim_params()

# overdrive parameter (boost brightness)
#TODO in parameter file?
img_overdrive = 2.5


# load image structure function
IA = ImageStructureFunction.fromFile(sim.ISE_outfpn)
Ni = IA.ISF.shape[0] 
    
vmx = IA.ISF.max() / img_overdrive # avoid autoscale of colormap

vf_redraw_init=False
vf_redraw_img=None
def vf_redraw(fri, ax):
    global ISFob
    global vmx
    global vf_redraw_init
    global vf_redraw_img
    if not vf_redraw_init:
        vf_redraw_img=ax.imshow(IA.ISF[fri], 
                                vmin = 0.0, vmax = vmx, 
                                origin = 'lower', animated = True)
        vf_redraw_init=True
    else:
        vf_redraw_img.set_array(IA.ISF[fri])

print("[ENTER]: toggle pause/play")
print("[LEFT]/[RIGHT]: scroll frames")
print("[MOUSE]: manipulate time bar")

videofig(Ni, vf_redraw, play_fps=10)


