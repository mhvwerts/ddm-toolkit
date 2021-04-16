#!/usr/bin/env python3
# coding: utf-8
#
# Cycle =
#      simulate Brownian motion, generate synthetic video frames,
#      analyze with DDM algorithm, fit Brownian model
#
# Repeat this cycle multiple times. For testing, statistics etc.
#
# Tuned for compatibility with Jupyter Notebook/Lab, Google Colab
# compatibility, enabling remote/cloud computing
#
# by Pierre Galloo-Beauvais and Martinus Werts, 2021


import time
import os, os.path
import pickle
import logging

import numpy as np

from scipy.optimize import curve_fit
from ddm_toolkit.simulation import brownian_softbox, random_coordinates
from ddm_toolkit.simulation import imgsynth2
from ddm_toolkit import ImageStructureEngineSelector
from ddm_toolkit import ImageStructureFunction
from ddm_toolkit import ISFanalysis_simple_brownian
from ddm_toolkit import sim_params, sim_params_empty



outputfolder = './datafiles/'



Nruntimeerror=0
Nassertionerror=0
def full_cycle(sim):
    global Nruntimeerror
    global Nassertionerror
   
    ####################################
    ####################################
    ### Simulate & synthesize video  ###
    ####################################
    ####################################

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
    for it in range(sim.Nt):
        img = imgsynth2(x1[:,it], y1[:,it], sim.img_w,
            -sim.img_border, -sim.img_border, 
            sim.bl_x+sim.img_border, sim.bl_y+sim.img_border,
            sim.img_Npx, sim.img_Npx,
            subpix = 2)
        ims.append(img)

    ##############################################
    ##############################################
    # CALCULATE VIDEO (IMAGE) STRUCTURE FUNCTION #
    ##############################################
    ##############################################
    
    ImageStructureEngine = ImageStructureEngineSelector(sim.ISE_type)
    # process using ImageStructureEngine
    ISE1 = ImageStructureEngine(sim.ISE_Npx, sim.ISE_Nbuf)
    for it in range(sim.Nframes):
        ISE1.push(ims[it])
    ISE1.ISFcount
        


    ###################################
    ###################################
    ### Fit Brownian model result   ###
    ###################################
    ###################################
    # get image structure function & apply REAL WORLD UNITS!
    IA = ImageStructureFunction.fromImageStructureEngine(ISE1)
    IA.real_world(sim.um_p_pix, sim.s_p_frame)
    # ===================================
    # FIT THE BROWNIAN MODEL
    # ===================================
    #
    # Initial guess
    D_guess = sim.D_guess

    # perform analysis of the ISF using the simple Brownian model
    # TODO: clean up
    #       remove try except
    #       error handling in ISFanalysis_simple_brownian
    #  e.g. return a "fit unsuccessful" status if something wrong with fitting
    #       The typical problems are known:
    #       "AssertionError" is usually due to too small a range of 'fittable'
    #        q's.
    #       "RuntimeError" is due to fits occasionnally not converging (choosing
    #       a better initial guess is likely to help)
    #       => the "fit unsuccessfull" may contain this info
    try:
        fitres = ISFanalysis_simple_brownian(IA, D_guess)
    except RuntimeError:
        Nruntimeerror+=1
        fitres = 'RuntimeError: probably one of the fits in the Brownian '\
                 'analysis did not converge. Using a better initial guess'\
                 ' may help.'
    except AssertionError:
        Nassertionerror+=1
        fitres = "AssertionError: probably the range of 'fittable' q's is too "\
                 "small, such that we may not obey q_low < q_opt < q_high. See "\
                 "'analysis.py' inside the toolkit..."          
    return (ims, ISE1, IA, fitres)



#Get Process ID (for creating intelligent output filenames)
pid = os.getpid()
print('Process = 0x{0:x}'.format(pid))

logname = os.path.join(outputfolder, 'x{0:x}_log.txt'.format(pid))
# following requires Python 3.9
#logging.basicConfig(filename=logname, encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(filename=logname, level=logging.DEBUG)
# TODO: somehow 'flush' the log buffer more often, so that we can
# view the log file more interactively.
logging.info('Logging activated... starting with a test of logging...')
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
logging.info('Logging test successful!')

#Get the starting time
start_time=time.time()
logging.info('Start simulation program at: '+time.ctime(start_time))

#Here, modify the parameters of the simulation
testing = False
if not testing:
    # Pierre Galloo-Beauvais' original program
    lesT=[1000,5000,10000,25000,50000,100000]
    lesbl=[200]
    lesNt=[400,600,800]
    lesD=[0.1,1,3,5,10]
    Nboucle = 3 #int(input("How many loops ? "))
    ## 'careful' error-free(?) program
    # lesT=[1000,5000,10000,25000,50000,100000]
    # lesbl=[200]
    # lesNt=[400,600,800]
    # lesD=[0.1] # 1.0] # value = 3.0 gives unexpected q ordering
    # Nboucle = 3 #int(input("How many loops ? "))
else:
    lesT=[1000.]
    lesbl=[200.]
    lesNt=[300]
    lesD=[0.1]
    Nboucle = 1

# 
# INITIALIZE SIMULATION/ANALYSIS PARAMETERS
# 
# Usually, we initialize like this:
# sim = sim_params()
# Here, we want to make sure that we have set all parameters by hand
# and that we have a minimal working set of parameters, so we start
# with a basically empty 'sim' instance
sim = sim_params_empty()

# Pierre's basic simulationparameters
sim.D = 0.1
sim.Np = 200
sim.bl = 200.
sim.bl_x = sim.bl
sim.bl_y = sim.bl
sim.Nt = 500
sim.T = 5000.
sim.dt = sim.T/sim.Nt
sim.img_border = 16.0
sim.img_w = 2.0
sim.img_Npx = 256
sim.ISE_Npx = sim.img_Npx
sim.img_l = (sim.bl + 2*sim.img_border)
sim.um_p_pix = sim.img_l/sim.img_Npx
sim.s_p_frame = sim.dt
sim.ISE_type = 0
sim.ISE_Nbuf = 200 if not testing else 40 
# watch out: if Nbuf too small, unexpected errors may occur with Brownian 
# fitting algorithm
sim.D_guess = 1.1

#Just the initialization of some parameters
lesAtot,lescompt=[0]*128,[0]*128

k1=1
Nruntimeerror,Nassertionerror,Nunknownerror=0,0,0
Ntotit=len(lesT)*len(lesbl)*len(lesNt)*len(lesD)*Nboucle

print('The program is going to do ', Ntotit, ' iterations.')

#Beginning of the loops
#
#TODO: 
# Put h, i, m, j in one-dimensional arrays first.
# The index of the elements is the iteration number
# This enables to re-start the program at a certain
# iteration after a crash/unexpected termination.
#
for h in range(Nboucle):
    for m in lesNt:
        for i in lesT:
            for j in lesD:
                sim.T = i
                sim.Nt = m
                sim.Nframes = m
                sim.D = j
                sim.D_guess = j * 1.13 # magic number to generate an
                                       # initial guess that is slightly
                                       # off the expected value
                print('k1 = ', k1, 'T = ',i,'; D_sim = ',j,
                      '; Nt = ',m,'; iteration n°',k1,'/',Ntotit)
                logging.info(time.ctime()+': iteration {0:d}'.format(k1))
                ims, ISE1, IA, fitres = full_cycle(sim)
                if type(fitres) == str:
                    logging.warning('Some fit error occured.')
                    print('***ERROR***')
                    print(fitres)
                    print('***********')
                    logging.info('Successfully caught error.')
                else:
                    D_fit = fitres['D_fit']
                    D_fit_ci95 = fitres['D_fit_CI95']
                    print('    D_fit = ', D_fit)
                print()
                fname = 'x{0:x}_{1:05d}.pkl'.format(pid, k1)
                fpname = os.path.join(outputfolder, fname)
                with open(fpname, 'wb') as f1:
                    pickle.dump(sim, f1)
                    pickle.dump(fitres, f1)
                k1+=1

logging.info('SUCCESS at '+time.ctime())
logging.info('Nruntimeerror = {0:d}'.format(Nruntimeerror))
logging.info('Nassertionerror = {0:d}'.format(Nassertionerror))
print("Total time : ", (time.time() - start_time)/60, ' minutes.')
print('Errors : ', Nruntimeerror + Nassertionerror, ', of which RunTimeError : ',
      Nruntimeerror, ' and AssertionError : ', Nassertionerror, '.')

##########################################
########END OF SIMULATIONS################
##########################################

