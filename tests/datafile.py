import os

from sys import argv
import numpy as np
from utils.tqdm import tqdm
from ddm_toolkit.simulation import brownian_softbox, imgsynth2

def generate_datafile(fname):
    # STEP 1: Brownian simulation and Video synthesis
    # ==============================
    # SIMULATION/ANALYSIS PARAMETERS
    # ==============================

    # SIMULATION parameters
    # D  [µm2 s-1]  Fickian diffusion coefficient of the particles
    # Np []         number of particles
    # bl [µm]       length of simulation box sides (square box)
    # Nt []         number of time steps => number of frames
    # T  [s]        total time
    D = 0.1
    Np = 200

    bl = 200.
    bl_x = bl     #Simulation box side length in x direction [µm]
    bl_y = bl

    Nt = 300
    T = 400.


    # IMAGE SYNTHESIS parameters
    # img_center [µm, µm]   NOT YET USED: coordinates of the center of the image
    # img_border [µm]       width of border around simuation box (may be negative!)
    # img_w      [µm]       width parameter of 2D Gaussian to simulate optical transfer function
    # img_Npx    []
    img_border = 16.
    img_w = 2.
    img_Npx = 256


    # IMAGE STRUCTURE ENGINE PARAMETERS
    # ISE_Nbuf []    buffer size of image structure engine
    # ISF_fpn        file (path) name for storing/retrieving image structure function
    ISE_Nbuf = 100
    ISE_Npx = img_Npx # frame size: Npx by Npx  must be equal to img_Npx
    ISF_fpn = 'datafiles/imageseq_pytest_tests_ISF.npz'

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
    np.savez_compressed(fname, img=ims)



def get_datafilename():
    fname = 'datafiles/imageseq_pytest_tests.npz'
    if os.path.isfile(fname):
        print('file with test image sequence exists')
    else:
        generate_datafile(fname)
    return fname
        

if (__name__=='__main__'):
    print(get_datafilename())
    
