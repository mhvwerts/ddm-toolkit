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
# STEP 4: Inspect ImageStructureFunction (videofig player)
#

from sys import argv
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit import ImageStructureFunction

from utils.videofig import videofig 


# ==============================
# SIMULATION/ANALYSIS PARAMETERS
# ==============================
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


# SIMULATION parameters
# D  [µm2 s-1]  Fickian diffusion coefficient of the particles
# Np []         number of particles
# bl [µm]       length of simulation box sides (square box)
# Nt []         number of time steps => number of frames
# T  [s]        total time
D = float(params['simulation']['D'])
Np = int(params['simulation']['Np'])

bl = float(params['simulation']['bl'])
bl_x = bl     #Simulation box side length in x direction [µm]
bl_y = bl

Nt = int(params['simulation']['Nt'])
T = float(params['simulation']['T'])


# IMAGE SYNTHESIS parameters
# img_center [µm, µm]   NOT YET USED: coordinates of the center of the image
# img_border [µm]       width of border around simuation box (may be negative!)
# img_w      [µm]       width parameter of 2D Gaussian to simulate optical transfer function
# img_Npx    []
img_border = float(params['imgsynth']['img_border'])
img_w = float(params['imgsynth']['img_w'])
img_Npx = int(params['imgsynth']['img_Npx'])


# IMAGE STRUCTURE ENGINE PARAMETERS
# ISE_Nbuf []    buffer size of image structure engine
# ISF_fpn        file (path) name for storing/retrieving image structure function
ISE_Nbuf = int(params['ISFengine']['ISE_Nbuf'])
ISE_Npx = img_Npx # frame size: Npx by Npx  must be equal to img_Npx
ISF_fpn = params['ISFengine']['ISF_fpn']

# conversion units, derived from simulation settings
img_l = (bl + 2*img_border)
um_p_pix = img_l/img_Npx
dt=T/Nt  # frame period [s]
s_p_frame = dt

# overdrive parameter (boost brightness)
#TODO in parameter file?
img_overdrive = 2.5


# load image structure function
IA = ImageStructureFunction.fromfilename(ISF_fpn)

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


