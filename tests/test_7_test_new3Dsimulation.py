# -*- coding: utf-8 -*-
"""
Test 7. Test (and illustrate) new-generation Brownian codes

in ddm_toolkit.simulation

"""

#### TODO: improve pytest integration, disable plotting when in pytest?
#### 


import pytest

from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt

# somewhat clumsy try...except imports
# to enable these test scripts to be run independently from pytest
# for example using spyder
try:
    import ddm_toolkit
except:
    import sys
    sys.path.append('./..')
    import ddm_toolkit

from ddm_toolkit.params import DDMParams
from ddm_toolkit.simulation import PRNG ## !!! be sure to use only a single PRNG
from ddm_toolkit.simulation import RunBrownPeriodicBox
from ddm_toolkit.simulation import ParticleSim2D
from ddm_toolkit.simulation import ImageSynthesizer2D
from ddm_toolkit.ddm import ImageStructureEngine
from ddm_toolkit.ddm import ImageStructureFunction
from ddm_toolkit.analysis import ISFanalysis_simple_brownian


# for pytest (very basic test...)
script_complete = False

print('')
print('')
print('7. Test new-generation Brownian simulation code')
print('')


# set up simulation parameters
params = DDMParams() 

params.sim_Np = 100
params.sim_bl = 200. # let's say this is in µm
params.sim_Nt = 300
params.sim_T = 600. # in seconds
params.sim_D = 0.4 # in  µm2 s-1

params.sim_img_border = 16.0
params.sim_img_w = 2.0
params.sim_img_Npx = 256
params.sim_img_I_offset = 0.06
params.sim_img_I_noise = 0.003

params.update_simulation_parameters()


##############################
# Simulate 3D Brownian motion and syntesize 2D image stacks
#   for now the 2D is just achieved by projecting all particles in a
#    2D plane (simple flatten, i.e. forget about one coordinate)
# (could later be put in a 'workflow' function)
#


##############################
# 3D brownian simulation
##############################

print("performing 3D brownian simulation...")

# initialize simulation parameters
Npart = params.sim_Np
Ntime = params.sim_Nt
Ndim = 3
boxsize = (params.sim_bl, params.sim_bl, params.sim_bl) 
D = params.sim_D
dt = params.sim_T / params.sim_Nt

# set up the storage for all particle coordinates
ptrack = np.zeros((Npart, Ntime, Ndim))

# initial condition (evenly distributed particles)
for idim in range(Ndim):
    ptrack[:, 0, idim] =  PRNG.random(Npart) * boxsize[idim]

# normally distributed noise
norm_noise = PRNG.normal(size = (Npart, Ntime-1, Ndim))

# run the actual simulation
RunBrownPeriodicBox(ptrack, norm_noise, D, dt, boxsize)

###########################
# end of 3D brownian simulation
##############################


#######################################################################
# Initialze a 2D ParticlSim2D object from the 3D coordinates
# We simply select 2 out of 3 dimensions, i.e. project all particles on
# a 2D surface
#
# TODO we could do 3 permutations of the projection (xy yz xz)
#      i.e. 0 1 ; 1 2; 2 0
#      or even 6, why not?
# Here we choose 1 2
#
psimul = ParticleSim2D(params,
                        x1 = ptrack[:, :, 1],
                        y1 = ptrack[:, :, 2]
                        )
#######################################################################


###########################
# image synthesis (code from 'simul1_make_simulated_image_stack.py')
#############################
print("creating simulated image stack...")
imgsynthez = ImageSynthesizer2D(psimul)
Nframes = params.sim_Nt
Npx = params.sim_img_Npx
ims = np.zeros((Nframes,Npx,Npx),
               dtype = np.float64)
for ix in tqdm(range(Nframes)):
    img = imgsynthez.get_frame(ix)
    ims[ix,:,:] = img


print("")
print('')

##################################
# DDM analysis of simulated image stack
#   code from 'simul3_calculate_ISF.py'

# set ISF and analysis  parameters
params.ISE_Npx = params.sim_img_Npx # use image width to set ISE width
params.ISE_Nbuf = 50
params.initial_guess_D = params.sim_D * 4.3 # choose an initial guess that is a factor 4.3 off

params.ISE_type = 6 # this is the fastest non-GPU DDM engine (requires numba)


ISE1 = ImageStructureEngine(params.ISE_Npx, params.ISE_Nbuf,
                            engine_model = params.ISE_type)

for it in tqdm(range(params.Nframes)):
    ISE1.push(ims[it])

ISF1 = ImageStructureFunction.fromImageStructureEngine(ISE1)

# set real world units
ISF1.real_world(params.um_p_pix, params.s_p_frame)

#
# DO FIT
result = ISFanalysis_simple_brownian(ISF1, params.initial_guess_D)

result.print_report()

print('')
print(60*'*')
print('  D_sim = ', params.sim_D)
print('  D_fit = ',result.D_fit)
print(60*'*')

## PLOTTING: 
# TODO perhaps deactivate plotting when running pytest,
#             and only execute when not in pytest
#

# plot radially averaged ISF
result.show_ISF_radavg()

# plot fits
result.show_fits()
result.show_Aq_Bq_kq()

# to show brownian video...
# see the code in 'simul2_inspect_video.py'


# for pytest (very basic test...)
script_complete = True


def test_3Dsimul_analysis_complete():
    assert script_complete

# 
# keep for reference, and to remember that pytest has more infrastructure for testing 
#
# @pytest.mark.parametrize("i_model", i_list)
# def test_ISEngine_run(i_model):
#     assert allcloses[i_model]
    


