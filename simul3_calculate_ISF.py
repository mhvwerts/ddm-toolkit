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

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit.tqdm import tqdm
from ddm_toolkit import ImageStructureEngineSelector
from ddm_toolkit import ImageStructureFunction
from ddm_toolkit import sim_params


#
# GET SIMULATION/ANALYSIS PARAMETERS
#
sim = sim_params()
ImageStructureEngine = ImageStructureEngineSelector(sim.ISE_type)


# CALCULATE VIDEO (IMAGE) STRUCTURE FUNCTION

# LOAD VIDEO
# video was saved using: np.savez_compressed(videof, img=ims)
ims = np.load(sim.vidfpn)['img']

# process using ImageStructureEngine
ISE1 = ImageStructureEngine(sim.ISE_Npx, sim.ISE_Nbuf)
for it in tqdm(range(sim.Nframes)):
    ISE1.push(ims[it])
ISE1.ISFcount

# write result file
ISE1.save(sim.ISE_outfpn)

#
# quick plot of radially average image structure function
#
IA = ImageStructureFunction.fromImageStructureEngine(ISE1)
IAqtau = np.zeros((len(IA.tauf),len(IA.u)))
for i in range(len(IA.tauf)):
    IAqtau[i,:] = IA.radavg(i)
    
plt.figure("radially averaged (video) image structure function")
plt.imshow(IAqtau)
