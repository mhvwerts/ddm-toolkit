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
# STEP 3: Calculate Image Structure Function from synthetic video

from sys import argv
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit.tqdm import tqdm

#from ddm_toolkit import ImageStructureEngine
#Here we use the experimental ImageStructureEngine3
#   
#TODO: incorporate into ddm_toolkit/__init__.py imports
from ddm_toolkit.ddm import ImageStructureEngine3 as ImageStructureEngine
#TODO: test against the original
#
# [2021-02-08]
#      simul3 : D (fit): 0.09861906880235917 µm2/s (+/-  0.0008906755653943241 , 95% CI)
#       1'34" runtime
#       'Original' ImageStructureEngine
#
#      simul3C: D (fit): 0.0986190689133698 µm2/s (+/-  0.0008906756662257405 , 95% CI)
#       1'00" runtime
#       This is only the FFT optimization, not the magnitude-squared optimization
#
#  NEW simul3C: D (fit): 0.09861906902831793 µm2/s (+/-  0.0008906755360855044 , 95% CI)
#       0'28" runtime
#       Both FFT and magnitude-squared optimizations
#

#
# very, very close, not identical ->  sign of different numerical approach
#  compare results further (compare full 2D ISFs)



from ddm_toolkit import ImageStructureFunction


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


# CALCULATE VIDEO (IMAGE) STRUCTURE FUNCTION

#RELOAD VIDEO
# video was saved using: np.savez_compressed(videof, img=ims)
ims = np.load(videof)['img']

#push onto DDM engine
ISF1 = ImageStructureEngine(ISE_Npx, ISE_Nbuf)
for it in tqdm(range(Nt)):
    ISF1.push(ims[it])
    #print('\r\tframe #{0:d}'.format(it), end='')
ISF1.ISFcount

ISF1.save(ISF_fpn)

IA = ImageStructureFunction.fromImageStructureEngine(ISF1)

IAqtau = np.zeros((len(IA.tauf),len(IA.u)))

for i in range(len(IA.tauf)):
    IAqtau[i,:] = IA.radavg(i)
    
plt.figure("radially averaged (video) image structure function")
plt.imshow(IAqtau)
