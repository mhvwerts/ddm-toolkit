#!/usr/bin/env python
# coding: utf-8
#
# development of a faster image synthesis algorithm
#
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from ddm_toolkit.simulation import random_coordinates
from ddm_toolkit.simulation import imgsynth1
from ddm_toolkit.simulation import imgsynth2

from utils.tabulate import tabulate

# ==============================
# SIMULATION/ANALYSIS PARAMETERS
# ==============================
N=500        #Number of time steps => number of frames
Np=200       #Number of particles
bl = 200.    # square simulation box 
bl_x = bl     #Simulation box length in x direction [µm]
bl_y = 200.     #Simulation box length in y direction [µm] 

img_border = 10.0 # width of image border around simulation box [µm]
img_Npx=512 # number of pixels in image
img_w = 5.0 # width of Gaussian spot in image [µm]

ISE_Nbuf = 100 # buffer size of image structure engine
ISE_Npx = img_Npx # frame size: Npx by Npx (square frames only)

# conversion units, derived from simulation settings
img_width = (bl + 2*img_border)
um_p_pix = img_width/img_Npx
# s_p_frame is in loop


### synthesize an image

# generate random x,y
x0 = random_coordinates(Np, bl_x)
y0 = random_coordinates(Np, bl_y)

img = imgsynth1(x0, y0, img_w,
    -img_border, -img_border, 
    bl_x+img_border, bl_y+img_border,
    img_Npx, img_Npx)

img2 = imgsynth2(x0, y0, img_w,
    -img_border, -img_border, 
    bl_x+img_border, bl_y+img_border,
    img_Npx, img_Npx)

img2s = imgsynth2(x0, y0, img_w,
    -img_border, -img_border, 
    bl_x+img_border, bl_y+img_border,
    img_Npx, img_Npx,
    subpix = 2)

plt.figure('Original imgsynth (sum of Gaussians) image')
plt.clf()
plt.imshow(img)
plt.colorbar()

plt.figure('Image from imgsynth2 without subpixels')
plt.clf()
plt.imshow(img2)
plt.colorbar()

plt.figure('Image from imgsynth2, subpix=2')
plt.clf()
plt.imshow(img2s)
plt.colorbar()

plt.pause(4.0)

totsq = np.sum(img**2)
sq2 = np.sum((img-img2)**2)
sq2s = np.sum((img-img2s)**2)

print('rel. sum over squares of differences ("squared error")')
print(tabulate([
        ["rel. squared error (no subpix)",    sq2/totsq],
        ["rel. squared error (subpix = 2)",   sq2s/totsq]
        ],floatfmt='e'))

def test_subpix2_better_than_no_subpix():
    assert (sq2/totsq > sq2s/totsq)

def test_subpix_rel_error_small():  
    assert (sq2/totsq < 1e-3)
    
    


