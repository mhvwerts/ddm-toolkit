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

import numpy as np

from ddm_toolkit.simulation import brownian_softbox, random_coordinates
from ddm_toolkit.simulation import imgsynth2
from ddm_toolkit.tqdm import tqdm

from ddm_toolkit import sim_params
       
        
# 
# GET SIMULATION/ANALYSIS PARAMETERS
#
sim = sim_params()


#
# SIMULATION (2D)
#
#set initial particle coordinates
x0 = random_coordinates(sim.Np, sim.bl_x)
y0 = random_coordinates(sim.Np, sim.bl_y)
#create array of coordinates of the particles at different timesteps
x1 = brownian_softbox(x0, sim.Nt, sim.dt, sim.D, sim.bl_x)
y1 = brownian_softbox(y0, sim.Nt, sim.dt, sim.D, sim.bl_y)

#
# make the synthetic image stack (video)
#
ims=[]
for it in tqdm(range(sim.Nt)):
    img = imgsynth2(x1[:,it], y1[:,it], sim.img_w,
        -sim.img_border, -sim.img_border, 
        sim.bl_x+sim.img_border, sim.bl_y+sim.img_border,
        sim.img_Npx, sim.img_Npx,
        subpix = 2)
    ims.append(img)

#
# save video
#
np.savez_compressed(sim.vidfpn, img=ims)
