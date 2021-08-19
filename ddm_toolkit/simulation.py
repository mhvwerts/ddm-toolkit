# -*- coding: utf-8 -*-
"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

simulation.py:
    Routines for simulation of simple Brownian motion of nanoparticles,
    generation of synthetic images of nanoparticle distributions, etc.
"""



import numpy as np
from numpy.random import Generator, PCG64
from scipy.ndimage import gaussian_filter


# Instantiate pseudo-random number generator (PRNG) for simulation/
# The PRNG is module-global, so that only one instance is used 
# throughout the simulation code
# 
# We might have used the well-known "Mersenne Twister" as provided 
# by numpy (MT19937), but we chose O’Neill’s "permuted 
# congruential generator" (PCG64). It is more recent (2014), and 
# reportedly has better statistical properties. MT19937 also gave 
# satisfactory results.
#
# Our use of PCG64 in simple Brownian simulations coupled to DDM analysis
# is actually one way of testing this PRNG for application in 
# Brownian dynamic simulation, since the expected result is perfectly
# known theoretically. (This assumes that the rest of the simulation program
# is perfect as well!)
#
#
# Bibliographic references:
#
# [1] @techreport{oneill:pcg2014,
#       title = "PCG: A Family of Simple Fast Space-Efficient Statistically Good 
#                Algorithms for Random Number Generation",
#       author = "Melissa E. O'Neill",
#       institution = "Harvey Mudd College",
#       address = "Claremont, CA",
#       number = "HMC-CS-2014-0905",
#       year = "2014",
#       month = Sep,
#       xurl = "https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf",
#     }
#
# [2] https://www.pcg-random.org/posts/history-of-the-pcg-paper.html
#
# [3] Bouillaguet, C.; Martinez, F.; Sauvage, J. "Predicting the PCG Pseudo-Random 
#     Number Generator In Practice." hal-02700791 (2020). 
#     https://hal.archives-ouvertes.fr/hal-02700791


PRNG = Generator(PCG64())


###############################################
# SECTION: Fundamental Brownian simulation code
###############################################

def random_coordinates(Np, d):
    """Generate the coordinates in 1D of a collection of Np
    evenly distributed particles.
    
    Parameters
    ----------
    Np : int
        Number of particles.
    d : float
        Size of the box. The generated coordinates will be in
        the range [0.0, d>

    Returns
    -------
    np.array
        1D array with Np particle coordinates.
    """
    return PRNG.random(Np)*d


def brownian_softbox(x0, Nt, dt, D, bl):
    """Generate time-sequences of 1D Brownian trajectories.
    
    Generates time-sequences of 1D Brownian trajectories of Np
    particles in a box with 'soft' boundaries, starting from 
    the initial coordinate of the particles that are supplied
    
    
    Parameters
    ----------
    x0 : np.array 
        1D array containing the initial coordinate for each particle
    """
        
    # generate Nt random steps for each particle in x0
    # these follow normal distribution around 0 with
    # with std dev delta*sqrt(dt)
    delta = np.sqrt(2*D)
    r = PRNG.normal(loc = 0.0, scale = delta*np.sqrt(dt),
                    size = x0.shape+(Nt,))
    
    # get number of particles from generated array
    Np = r.shape[0]
    assert Nt == r.shape[1], "what went wrong?"

    # Record the initial condition
    r[:,0]=x0[:]
    # Integrate, respecting the boundaries
    for i in range(Np):
        for j in range(1,Nt):
                k=r[i,j-1]+r[i,j] 
                if k <= bl and k>=0:
                    r[i,j]=k
                else:
                    r[i,j]=r[i,j-1]
    return r



###############################################
# SECTION: Fundamental image synthesis code
###############################################

def imgsynth1(px, py, w, x0, y0, x1, y1, Nx, Ny):
    """Create a synthetic digital image from particle coordinates.
    
    Each particle is represented as a 2D Gaussian of radial width w. The
    image consists of the sum of the Gaussians of each particle.
    
    Parameters
    ----------
    px : 1D ndarray float 
        Particle x coordinates (µm), i.e. px[i_particle].
    py : 1D ndarray float 
        Particle x coordinates (µm), i.e. px[i_particle].
    w : float
        Radial width of Gaussian (µm).
    x0, y0 : float, float
        Bottom left coordinates of viewport (µm).
    x1, y1 : float, float
        Top right coordinates of viewport (µm).
    Nx, Ny : int, int
        Number of pixel in x resp. y direction.


    Returns
    -------
    img : 2D ndarray
        Digital image of pixel intensities.
    
    
    Details
    -------
       
    The individual Gaussians are normalised (2D integral = 1), such that
    np.sum(img) is equal to number of particles (within numerical error)

    The pixel coordinates are the centers of each pixel, *i.e.* (0.0, 0.0) is
    the pixel with (0.0, 0.0) at its center. Depending on the choice of the
    viewport coordinates, one may or may not have (0.0, 0.0) as an actual pixel
    center. Note that the viewport coordinates are the edges of the 
    image. Therefore, these coordinates are positioned *half a pixel width*
    from the center of the edge pixel. This gives a coherent set of
    coordinates: *e.g.*, with bottom left (-10.24,-10.24) and 
    top right (10.24,10.24) and 256 pixels, the center coordinates of the bottom 
    left pixel will be (-10.20,-10.20), and the center (0,0) will be 
    *midway between* pixel numbers 127 and 128 (in the exact center of the
    image). The edge coordinates are then indeed as specified.

    The present implementation is relatively slow, but precise.

    """
 
    Np = len(px) # Np: number of particles
    assert len(py) == Np, 'px and py should have same number of elements'
    dx = (x1-x0)/Nx
    dy = (y1-y0)/Ny
    assert np.isclose(dx,dy), 'need square pixels! non-square pixels are not supported'
    
    img = np.zeros((Nx,Ny))
    
    #TODO add option to enable user to retrieve x, y scales and/or coordinates
    xco = (np.arange(Nx)+0.5)*dx + x0
    yco = (np.arange(Ny)+0.5)*dy + y0
    xc,yc = np.meshgrid(xco,yco) 
    #xc and yc contain the coordinates at the center of each pixel cell
    
    sss = 2.0*w**2
    h = (dx*dy)/(2*np.pi*w**2)
    #h normalizes the Gaussian (integral sum = 1)
    
    for ip in range(Np):
        #the following two lines calculate a 2D Gaussian
        # centered at the particle coordinates
        r2 = (xc-px[ip])**2 + (yc-py[ip])**2
        pimg = h*np.exp(-r2/sss)
        img += pimg
    return img



def img_rebin(img_in, Nbin):
    '''Re-bin a 2D image array.
    
    Each point in the resulting 'binned' array stores the SUM of points
    within square of Nbin x Nbin in the old array.
    We use the sum, since we want to conserve total intensity.

    Parameters
    ----------    
    img_in : 2D numpy array
        Input image array
    Nbin : 
        Number of pixels in each direction to combine into one bin. This
        is square: total pixels are Nbin x Nbin
    '''
    olds0 = img_in.shape[0]
    olds1 = img_in.shape[1]
    news0 = olds0//Nbin
    news1 = olds1//Nbin
    olds0 = news0 * Nbin
    olds1 = news1 * Nbin
    
    img = img_in[0:olds0,0:olds1]
    shape = (news0, Nbin,
             news1, Nbin)
    imgbin = img.reshape(shape).sum(-1).sum(1)
    return imgbin



def imgsynth2(px, py, w, x0, y0, x1, y1, 
              Nx: int, Ny: int, subpix: int = 1):
    '''
    Faster algorithm for image synthesis (compared to imgsynth1)
    Uses parameter set identical to imgsynth 

    Input parameters:
    1D array of particle x coordinates px[i_particle] (µm)
    1D array of particle y coordinates py[i_particle] (µm)
    radial width of Gaussian (µm)
    x0,y0: bottom left of viewport (µm)
    x1,y1: top right of viewport  (µm)
    Nx: number of x pixels
    Ny: number of y pixels
    subpix: calculate the image on a 'subpix'-times oversampled
            grid (e.g. subpix = 2 -> 512x512 image becomes
            1024x1024 for synthesis, then re-binned to 512x512)
            We recommend using odd numbers (1, 3, 5)
    
    Output:
    imgsynth2: 2D numpy array with pixel intensities
    
    This algorithm is based on Gaussian convolution. 
    
    There are small differences between the images calculated by
    'imgsynth2' and those from 'imgsynth' due to the fact that 
    imgsynth2 does not do 'subpixel' positioning.
    In imgsynth2, the central position of the Gaussian is necessarily
    at the center of the pixel, whereas in imgsynth, the calculation
    of the precisely positioned Gaussian is sampled onto the image
    pixels.
    Since the positional error in imgsynth2 is random, this is not
    expected to introduce a large error in the Brownian videos.
    If necessary, this situation may be improved upon by working on 
    an oversampled grid and then finally binning to the desired image
    dimension (e.g. 512x512 calculation for a final 256x256 image). This
    is achieved by using the 'subpix' keyword parameter.        
    '''
    assert subpix >= 1, 'subpix should be 1 or larger'
    Np = len(px) # Np: number of particles
    assert len(py) == Np, 'px and py should have same number of elements'
    dx = (x1-x0)/(Nx*subpix)
    dy = (y1-y0)/(Ny*subpix)
    assert np.isclose(dx,dy), 'need square pixels! non-square pixels are not supported'
    assert Nx==Ny, 'this implementation only works with square target images'
    um_p_pix = dx # improve readability

    # generate image with particles confined to indiv. pixels
    # attention x and y should be applied in exactly this order
    # to be compatible with imgsynth
    hist2d = np.histogram2d(py, px, bins=Nx*subpix,
                        range=[[y0, y1],
                               [x0, x1]]) 
    img2r = hist2d[0]

    # sigma for Gaussian filter, based on w from imgsynth
    # conversion
    sigma = w/um_p_pix    
    img2hi = gaussian_filter(img2r, sigma) 
    if (subpix > 1):
        img2 = img_rebin(img2hi, subpix)
    else:
        img2 = img2hi
    return img2



####################################################
# SECTION: Simulation interface classes
####################################################

class ParticleSim2DBrownian:
    """Particle simulator class (2D Brownian motion)
    
    Takes DDMparameter object as input, generates particle trajectories
    
    Properties
    
    x1, y1 (np.array) => x resp. y coordinates ()
        
    """
    def __init__(self, ddmpars):
        
        # store reference to all parameters
        self.ddmpars = ddmpars
        
        #set initial particle coordinates
        x0 = random_coordinates(ddmpars.sim_Np, ddmpars.sim_bl)
        y0 = random_coordinates(ddmpars.sim_Np, ddmpars.sim_bl)
        #create array of coordinates of the particles at different timesteps
        self.x1 = brownian_softbox(x0, ddmpars.sim_Nt, ddmpars.sim_dt,
                                       ddmpars.sim_D, ddmpars.sim_bl)
        self.y1 = brownian_softbox(y0, ddmpars.sim_Nt, ddmpars.sim_dt,
                                       ddmpars.sim_D, ddmpars.sim_bl)
        
    def get_coordinates2D(self, ix):
        """return particle coordinates at frame(time) index ix
        
        returns
        
        a tuple x, y of vectors containing sim_Npart coordinates  
        """
        return (self.x1[:,ix], self.y1[:,ix])
        
    def stream_coordinates2D(self):
        """iterator: streams particle coordinates, one time-step at a time
        
        can only iterated once!
        
        typical use case
        
        # psimul is an instance of ParicleSim2DBrownian
        for x, y in psimul.stream_coordinates2D():
            # x contains the x coordinates of all particles of the present time step
            # y contains the y coordinates of all particles of the present time step
            pass
            
        """
        for ix in range(self.ddmpars.sim_Nt):
            yield (self.x1[:,ix], self.y1[:,ix])



class ImageSynthesizer2D:
    """Synthesize images/stream of images from particle coordinates
    
    TODO: should also choose the exact imgsynthesis algorithm etc.
    """
    def __init__(self, particle_simulator):
        self.get_coordinates = particle_simulator.get_coordinates2D
        self.coordstream = particle_simulator.stream_coordinates2D()
        
        ddmpars = particle_simulator.ddmpars
        # transfer all parameters
        self.ddmpars = ddmpars
        
        # essential parameters
        self.Nframes = ddmpars.Nframes
        
        # extract parameters for synthesis
        self.sim_img_w = ddmpars.sim_img_w
        self.sim_img_border = ddmpars.sim_img_border
        self.sim_bl = ddmpars.sim_bl
        self.sim_img_Npx = ddmpars.sim_img_Npx

        try:
            self.sim_img_I_offset = ddmpars.sim_img_I_offset
        except AttributeError:
            self.sim_img_I_offset = None # None means no offset to be applied
        
        try:
            self.sim_img_I_noise = ddmpars.sim_img_I_noise
        except AttributeError:
            self.sim_img_I_noise = -1.0 # negative noise means no noise

    
    def make_imgframe(self, px, py):
        """generate a synthetic image from vectors of particle coordinates
        
        applies the parameters defined in the ddmpars of particle_simulator object
        (i.e. the parameters are transmitted from the particle simulator to the
        image synthesizer, only the particle simulator takes a specific DDMparameter object)
        
        """
        #TODO choice between imgsynth1 and imgsynth2 (in __init__)
        #     and tune subpix value for imgsynth2
        img = imgsynth2(px, py, 
                         self.sim_img_w,
                         -self.sim_img_border, -self.sim_img_border, 
                         self.sim_bl+self.sim_img_border, self.sim_bl+self.sim_img_border,
                         self.sim_img_Npx, self.sim_img_Npx,
                         subpix = 2)
        if not (self.sim_img_I_offset is None):
            img += self.sim_img_I_offset
        if not (self.sim_img_I_noise <= 0.0):
            imgnoise = PRNG.normal(loc = 0.0, scale = self.sim_img_I_noise,
                                   size = (self.sim_img_Npx, self.sim_img_Npx))
            img += imgnoise 
        return img
    
    def get_frame(self, ix_frame):
        x, y = self.get_coordinates(ix_frame)
        return self.make_imgframe(x, y)
        
    def stream_frames(self):
        for x, y in self.coordstream:
            yield self.make_imgframe(x, y)

