# -*- coding: utf-8 -*-
"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

parameters.py:
    classes to load and store DDM simulation and processing parameters
"""

from sys import argv
from configparser import ConfigParser

class sim_params:
    """
    Read parameter file as specified in argument to script call (argv)
    Read from "simul0_default_params.txt" if nothing in argv
    Subsequently populate all available simulation parameters.
   

    SIMULATION parameters
    
    D          [µm2 s-1]  Fickian diffusion coefficient of the particles
    Np         []         number of particles
    bl         [µm]       length of simulation box sides (square box)
    Nt         []         number of time steps => number of frames
    T          [s]        total time        


    SIMULATION IMAGE SYNTHESIS parameters
    
    img_center [µm, µm]   NOT YET USED: coordinates of the center of the image
    img_border [µm]       width of border around simuation box (may be negative!)
    img_w      [µm]       width parameter of 2D Gaussian to simulate
                          optical transfer function
    img_Npx    []         width and height of output image
    img_I_offset []       apply a DC offset to the pixel intensity
    img_I_noise []        apply a Gaussian noise to the pixel intensity. This
                          parameter is the standard deviation of the Gaussian
                          noise distribution ('scale' parameter in random.normal)
                          Set this to a negative value if no noise generation is
                          needed, or remove this parameter altogether.
                          (The integrated intensity of each particle is fixed
                           at 1.0. Also, the intensity is a float32 value.)
    img_file              file (path) name for storing video stack


    VIDEO INSPECTION/PLAY BACK ('ANIMATION') parameters

    Nview        []       number of frames to play back (-1 means all frames)

    
    IMAGE STRUCTURE ENGINE parameters

    ISE_type              select type of ImageStructureEngine
                          (0 is basic reference engine)
    ISE_Nbuf     []       buffer size of image structure engine
    ISF_outfpn            file (path) name for storing/retrieving image
                          structure function
    
    
    SIMPLE BROWNIAN ANALYSIS parameters
    
    D_guess    [µm2 s-1]   Initial guess of diffusion coefficient for analysis of
                           the DDM image structure function using the simple
                           Brownian model

     
    CONVERSION UNITS and DERIVED parameters
    
    dt = T/Nt                  simulation time step
    s_p_frame = dt             (simulated) video frame period (1 / frm rate)
    um_p_pix = img_l/img_Npx   (simulated) video pixel step (1 / resolution)
    Nframes = Nt               number of frames in the (simulated) video
    """
    def __init__(self):
        default = "simul0_default_params.txt"

        params = ConfigParser()
        argc = len(argv)
        if argc == 1:
            parfn = default
        elif argc == 2:
            parfn = argv[1]
        else:
            raise Exception('invalid number of arguments')
        params.read(parfn)

        self.D = float(params['simulation']['D'])
        self.Np = int(params['simulation']['Np'])
        
        self.bl = float(params['simulation']['bl'])
        self.bl_x = self.bl     #Simulation box side length in x direction [µm]
        self.bl_y = self.bl
        
        self.Nt = int(params['simulation']['Nt'])
        self.T = float(params['simulation']['T'])

        self.img_border = float(params['imgsynth']['img_border'])
        self.img_w = float(params['imgsynth']['img_w'])
        self.img_Npx = int(params['imgsynth']['img_Npx'])
        try:
            self.img_I_offset = float(params['imgsynth']['img_I_offset'])
        except KeyError:
            self.img_I_offset = None
        try:
            self.img_I_noise = float(params['imgsynth']['img_I_noise'])
        except KeyError:
            self.img_I_noise = -1.0
        self.vidfpn = params['imgsynth']['vidfpn']

        try:
            self.ISE_type = int(params['ISFengine']['ISE_type'])
        except KeyError:
            self.ISE_type = 0
        self.ISE_Nbuf = int(params['ISEngine']['ISE_Nbuf'])
        self.ISE_Npx = self.img_Npx
        self.ISE_outfpn = params['ISEngine']['ISE_outfpn']        
 
        self.Nview = int(params['animation']['Nview'])        

        self.D_guess = float(params['analysis_brownian']['D_guess'])

        img_l = (self.bl + 2*self.img_border)
        self.um_p_pix = img_l/self.img_Npx
        self.dt = self.T/self.Nt
        self.s_p_frame = self.dt      
        self.Nframes = self.Nt
        
        
    def params_exist(self):
        """Dummy function used in testing for existence of 
        sim_params class object
        
        Use case:
                
        try:
            # IPython: use '%run -i' and tune attributes of sim
            sim.params_exist()
        except (NameError, AttributeError) as combinerr:
            print("Loading parameters from file")
            sim = sim_params()
            
        

        Returns
        -------
        None.

        """
        pass
