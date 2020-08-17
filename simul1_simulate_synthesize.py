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
# STEP 1: Brownian simulation and Video synthesis

from sys import argv
from configparser import ConfigParser

import numpy as np

from utils.tqdm import tqdm

from ddm_toolkit.simulation import brownian_softbox, imgsynth2



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
videof = params['imgsynth']['img_file']


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


# SIMULATION

#set initial particle coordinates
x0=np.random.random(Np)*bl_x
y0=np.random.random(Np)*bl_y
#create array of coordinates of the particles at different timesteps
x1=brownian_softbox(x0, Nt, dt, D, bl_x)
y1=brownian_softbox(y0, Nt, dt, D, bl_y)

#make the synthetic image stack
ims=[]
for it in tqdm(range(Nt)):
    img = imgsynth2(x1[:,it], y1[:,it], img_w,
        -img_border, -img_border, 
        bl_x+img_border, bl_y+img_border,
        img_Npx, img_Npx,
        subpix = 2)
    ims.append(img)

#save video
np.savez_compressed(videof, img=ims)
