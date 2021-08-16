# -*- coding: utf-8 -*-
"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

parameters.py:
    classes to load and store DDM simulation and processing parameters
"""
from sys import argv
from configparser import ConfigParser



class DDMParams:
    """New version of unified DDM Toolkit parameter class
    
    Documentation to be updated from ddm_toolkit.parameters etc.
    
    in its basic version, it generates an empty parameter object
    to be filled with the necessary parameters
    
    """
    @classmethod
    def fromSimulationConfigFile(cls, fpn):
        """
        This returns a DDMParams object with parameters
        populated using a configuration file for simulated data
        """    
        CP = ConfigParser()
        CP.read(fpn)
        #Create and POPULATE class
        params = cls()
        params.configfile = fpn
        params.populate_from_configparser_sim(CP)
        return params
    
    @classmethod
    def defaultSimulationParams(cls):
        """
        Generate a DDMParams object filled with the 'default'
        simulation parameters


        Returns
        -------
        Instance of DDMParams class, populated with default simulation
        parameters

        """
        global default_sim_params
        
        CP = ConfigParser()
        CP.read_string(default_sim_params)
        #Create and POPULATE class
        params = cls()
        params.populate_from_configparser_sim(CP)
        return params
    
    def populate_from_configparser_sim(self, CP):
        self.sim_D = float(CP['simulation']['D'])
        self.sim_Np = int(CP['simulation']['Np'])
        
        self.sim_bl = float(CP['simulation']['bl'])

        self.sim_Nt = int(CP['simulation']['Nt'])
        self.sim_T = float(CP['simulation']['T'])

        self.sim_img_border = float(CP['imgsynth']['img_border'])
        self.sim_img_w = float(CP['imgsynth']['img_w'])
        self.sim_img_Npx = int(CP['imgsynth']['img_Npx'])
        self.img_Npx = self.sim_img_Npx
        try:
            self.sim_img_I_offset = float(CP['imgsynth']['img_I_offset'])
        except KeyError:
            self.sim_img_I_offset = None
        try:
            self.sim_img_I_noise = float(CP['imgsynth']['img_I_noise'])
        except KeyError:
            self.sim_img_I_noise = -1.0
        self.vidfpn = CP['imgsynth']['vidfpn']


        # videoplayer settings (TODO: document)
        video_overdrive = 1.0
        try:
            video_overdrive = float(CP['videoplayer']['video_overdrive'])
        except KeyError:
            pass
        self.video_overdrive = video_overdrive

        self.video_Nview = int(CP['videoplayer']['Nview'])   

        try:
            self.ISE_type = int(CP['ISFengine']['ISE_type'])
        except KeyError:
            self.ISE_type = 0
        self.ISE_Nbuf = int(CP['ISEngine']['ISE_Nbuf'])
        self.ISE_Npx = self.img_Npx
        self.ISE_outfpn = CP['ISEngine']['ISE_outfpn']        
        
        # in certain cases, the ISF has been radially averaged
        # before saving, in order to save space
        # in that case, ISF_radialaverage is set to True
        self.ISF_radialaverage = False
        try:
            if CP['ISFengine']['ISF_radialaverage']=='True':
                self.ISF_radialaverage = True
        except KeyError:
            pass
        
        imgoverdrive = 2.1 # default value 
        try:
            imgoverdrive = float(CP['ISFengine']['ISF_display_overdrive'])
        except KeyError:
            pass        
        self.ISF_viewer_overdrive = imgoverdrive
 
     

        self.D_guess = float(CP['analysis_brownian']['D_guess'])

        # derived parameters
        self.sim_dt = self.sim_T/self.sim_Nt
        self.img_l = (self.sim_bl + 2*self.sim_img_border)
        self.um_p_pix = self.img_l/self.img_Npx
        self.s_p_frame = self.sim_dt      
        self.Nframes = self.sim_Nt
        
    
    def update_simulation_parameters(self):
        """update parameters derived from the simulation parameters

        for example, the simulation time step sim_dt is calculated from sim_T and sim_Nt

        only use this with simulations, not with real video
        
        TODO: this method should disappear (become transparent)
        """
        
        # TODO: the following should be automatic (copy to where sim_dt
        # is being used...)
        self.sim_dt = self.sim_T/self.sim_Nt
        
        # 'real world' video parameters
        self.s_p_frame = self.sim_dt      
        self.Nframes = self.sim_Nt
        self.img_Npx = self.sim_img_Npx
        self.img_l = (self.sim_bl + 2*self.sim_img_border)
        self.um_p_pix = self.img_l/self.img_Npx


def DDMParams_from_configfile_or_defaultpars():
    """
    Get parameters either from the default parameters (defined in this module)
    or from a config file (if specified as command line argument)

    Returns
    -------
    DDMParams object with DDM parameters
    

    """
    argc = len(argv)
    if argc == 1:
        print('Using default parameters from ddm_toolkit.parameters.')
        params = DDMParams.defaultSimulationParams()
    elif argc == 2:
        parfn = argv[1]
        print('Using parameters from file:', parfn)
        params = DDMParams.fromSimulationConfigFile(parfn)
    else:
        print('argc = ',argc)
        print('argv = ',argv)
        raise Exception('invalid number of arguments')
    return params


########
# SECTION: default parameters go here
#
## DEFAULT SIMULATION PARAMETERS 


default_sim_params = """# DEFAULT SIMULATION-ANALYSIS PARAMETERS (v210326)

[simulation]
# D  [µm2 s-1]  Fickian diffusion coefficient of the particles
# Np []         number of particles
# bl [µm]       length of simulation box sides (square box)
# Nt []         number of time steps = number of frames
# T  [s]        total simulation time
#
D = 0.1
Np = 200
bl = 200.
Nt = 500
T = 1000.

[imgsynth]
# img_border [µm]       width of border around simuation box (may be negative!)
# img_w      [µm]       width parameter of 2D Gaussian to simulate optical
#                       transfer function
# img_Npx    []         width and height of synthetic image in pixels
# img_I_offset []       apply a DC offset to the pixel intensity
# img_I_noise []        apply a Gaussian noise to the pixel intensity. This
#                       parameter is the standard deviation of the Gaussian
#                       noise distribution ('scale' parameter in random.normal)
#                       Set this to a negative value if no noise generation is
#                       needed, or remove this parameter altogether.
#                       (The integrated intensity of each particle is fixed
#                       at 1.0. Also, the intensity is a float32 value.)
# img_file              file (path) name for storing video stack
#
img_border = 16.0
img_w = 2.0
img_Npx = 256
img_I_offset = 0.06
img_I_noise = 0.03
vidfpn = datafiles/simul1_result_video.npz

[videoplayer]
# Nview   []  number of frames to play back (-1 means all frames)
#
video_overdrive = 1.7
Nview = -1

[ISEngine]
# ISE_type       select type of ImageStructureEngine (0 is basic reference engine)
# ISE_Nbuf []    buffer size of image structure engine
# ISE_outfpn        file (path) name for storing/retrieving image structure function
#
ISE_type = 0
ISE_Nbuf = 100
ISE_outfpn = datafiles/simul3_result_ISF.npz

[analysis_brownian]
# D_guess    [µm2 s-1]   Initial guess of diffusion coefficient for analysis of
#                        the DDM image structure function using the simple
#                        Brownian model
# In this example, we use a value that is deliberately off by a factor of 11
# from the simulation input value
#
D_guess = 1.1

"""





##########################################################
# SECTION: Legacy code (for compatibility with old scripts)
# TODO: adapt and remove


# 'SOFT' REMOVE (keep around just in case removal breaks something)
# TODO: clean-up
# =============================================================================
# class sim_params_empty:
#     def params_exist(self):
#         """Dummy function used in testing for existence of 
#         sim_params class object
#         
#         Use case:
#                 
#         try:
#             # IPython: use '%run -i' and tune attributes of sim
#             sim.params_exist()
#         except (NameError, AttributeError) as combinerr:
#             print("Loading parameters from file")
#             sim = sim_params()
#             
#         
# 
#         Returns
#         -------
#         None.
# 
#         """
#         pass
# =============================================================================
    

class sim_params:
    """
    Read parameter file as specified in argument to script call (argv)
    Use default parameters from global string variable 'default_sim_params'
    (end of this file), if no argument specified.
    
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
        global default_sim_params
        params = ConfigParser()
        argc = len(argv)
        if argc == 1:
            params.read_string(default_sim_params)
        elif argc == 2:
            parfn = argv[1]
            params.read(parfn)
        else:
            print('argc = ',argc)
            print('argv = ',argv)
            raise Exception('invalid number of arguments')
            
        print('WARNING! You are running legacy code... this needs to be updated!')


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
        
   
        
