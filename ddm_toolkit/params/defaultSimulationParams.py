default_sim_params = """# DEFAULT SIMULATION-ANALYSIS PARAMETERS (v220817)

[simulation]
# D  [µm2 s-1]  Fickian diffusion coefficient of the particles
# Np []         number of particles
# bl [µm]       length of simulation box sides (square box)
# Nt []         number of time steps = number of frames
# T  [s]        total simulation time
# v [µm.s^-1]        Flow velocity
# phi_0 [rad]   Flow orientation with respect to x-axis
#
D = 7.87
Np = 200
bl = 200.
Nt = 500
T = 100.
#v = 10
#phi = 0.

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
img_w = 1.3
img_Npx = 256
img_I_offset = 0.06
img_I_noise = 0.01
vidfpn = ../datafiles/simul1_video.npz

[videoplayer]
# Nview   []  number of frames to play back (-1 means all frames)
#
video_overdrive = 1.7
Nview = -1

[ISEngine]
# ISE_type       select type of ImageStructureEngine (0 is basic reference engine)
# ISE_Nbuf []    buffer size of image structure engine
# ISF_outfpn        file (path) name for storing/retrieving image structure function
#
ISE_type = 6
ISE_Nbuf = 100
ISF_outfpn = ../datafiles/simul3_ISF.npz

[analysis_brownian]
# D_guess    [µm2 s-1]   Initial guess of diffusion coefficient for analysis of
#                        the DDM image structure function using the simple
#                        Brownian model
# In this example, we use a value that is deliberately off by a factor of 11
# from the simulation input value
#
D_guess = 1.1

"""
