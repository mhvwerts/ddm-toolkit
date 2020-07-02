from time import time

from utils.tqdm import tqdm

import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.stats.distributions import t as student_t
from lmfit import Model

from ddm_toolkit.ddm import ImageStructureEngine



def brownian_softbox(x0, Nt, dt, D, bl):
    """"Generate time-sequences of 1D Brownian trajectories.
    
    Generates time-sequences of 1D Brownian trajectories of Np
    particles in a box with 'soft' boundaries, starting from 
    the initial coordinate of the particles that are supplied
    
    
    Parameters
    ----------
    x0 : np.array 1D containing the initial coordinate for each particle
    """
        
    delta = np.sqrt(2*D)
    r = norm.rvs(size=x0.shape+(Nt,), scale=delta*np.sqrt(dt))
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



def imgsynth1(px, py, w, x0, y0, x1, y1, Nx, Ny):
    '''
        Input parameters:
        1D array of particle x coordinates px[i_particle] (µm)
        1D array of particle y coordinates py[i_particle] (µm)
        radial width of Gaussian (µm)
        x0,y0: bottom left of viewport (µm)
        x1,y1: top right of viewport  (µm)
        Nx: number of x pixels
        Ny: number of y pixels
        
        Output:
        imgsynth: 2D numpy array with pixel intensities
        
        The individual Gaussians are normalised (2D integral = 1), such that
        np.sum(img) is equal to number of particles (within numerical error)

        The present implementation takes quite a while to calculate an image.
    '''
    Np = len(px) # Np: number of particles
    assert len(py) == Np, 'px and py should have same number of elements'
    dx = (x1-x0)/Nx
    dy = (y1-y0)/Ny
    assert np.isclose(dx,dy), 'need square pixels! non-square pixels are not supported'
    
    img = np.zeros((Nx,Ny))
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



# GENERAL utility functions

def closestidx(vec, val):
    """Find the index of the vector element closest to a value.
    
    Parameters
    ----------
    vec : 1D numpy vector
        Vector in which the value is searched.
    val : float
        Value to be found in vector.
        
    Returns
    -------
    int
        Index of element in `vec` with value closest to `value`
    """
    dist=abs(vec-val)
    return np.argmin(dist)



def conf95(stdev, Ndata, Npar):
    """Calculate (one half) of the symmetric 95% confidence interval.
    
    The symmetric 95% confidence interval is calculated from the standard
    deviation and number of degrees of freedom, using Student's t
    distribution.
    
    Parameters
    ----------
    stdev : float
        Standard deviation.
    Ndata : int
        Number of data points.
    Npar : int
        Number of parameters.
    
    Returns
    -------
    float
        The half-width of the 95% confidence interval, such that it
        can be reported in the tradtional +/- manner.
    
    based on:
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
    """
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)
    dof = max(0, Ndata - Npar)
    tval = student_t.ppf(1.0-alpha/2., dof)
    return (stdev*tval)


# ========================================================
# Functions for running tests 
#TODO (should be placed elsewhere)
# ========================================================

def F1(x0, N, dt, D1, bl_x, bl_y):
    #create array of coordinates of the particles at different timesteps
    x1=brownian_softbox(x0, N, dt, D1, bl_x)
    y1=brownian_softbox(y0, N, dt, D1, bl_y)


    w = 2.0
    border = 18.52
    Np1 = x1.shape[0]
    Nt1 = x1.shape[1]

    Npx=256

    #make the synthetic image stack
    Ni1=Nt1
    ims1=[]
    for it in tqdm(range(Ni1)):
        img = imgsynth(x1[:,it], y1[:,it], w,
            -border, -border, 200.0+border, 200.0+border,
                                   256, 256)
        ims1.append(img)
    

        #push onto DDM engine
        ISF_Nbuf = 100
        ISF_Npx = 256
        ISF1 = ImageStructureEngine(ISF_Npx, ISF_Nbuf)

        Ni=len(ims1[0])
        t0 = time()
        for it in range(Ni):
            ISF1.push(ims1[it])
            print('\r\tframe #{0:d}'.format(it), end='')
        t1 = time()
        ISF1.ISFcount
        outfp = 'test_ISF_compare.npz'
        ISF1.save(outfp)
        return
 
def F2():
        i1=len(IAqtau1[len(IA1.taus)-1])-1
        while IAqtau1[len(IA1.taus)-1,i1]*1.00<=0.10:
            i1=i1-1
        j1=i1
        while IAqtau1[len(IA1.taus)-1,j1]>=0.10 and j1>0:
            j1=j1-1

        #fit the curve to find the diffusion coefficient
        def Fitting_Function(t, A, B, D1):
                f= 1.00- np.exp(-D1*t)
                return A*f-B

        Function = Model(Fitting_Function)
        Function_params1 = Function.make_params(A=14, B=0.33, D1=0.0201 )
        q=15
        Parameters=np.zeros((i1-j1+1,3))
        qi=q
        
        for qi in range(len(Parameters)):
            FIT = Function.fit(IAqtau1[:,qi+j1], Function_params1 ,t=IA1.taus[:])
            Parameters[qi]= [FIT.best_values['A'], FIT.best_values['B'], FIT.best_values['D1']]
            Function_params1 = Function.make_params(A = FIT.best_values['A'], B = FIT.best_values['B'], D1 = FIT.best_values['D1'])
            qi = qi+1

        def Diff(q,d):
            return d*q**(2)
        diff_model1=Model(Diff)
        diff_param1=diff_model1.make_params(d=1.0)
        diff_fit1 = diff_model1.fit (Parameters[:,2], diff_param1 , q=IA1.qs[j1:i1+1])
        result1 = diff_model1.fit(Parameters[:,2], q=IA1.qs[j1:i1+1], d=1.0)
        d = diff_fit1.best_values['d']
        D_T = d / (4*np.pi ** 2)
        d1=D_T*(bl_x + 2.00*border)*(bl_y + 2.00*border)*N/(T*Npx*Npx)
        error=abs(D1-d1)/D1*100.00
        D.append([D1,d1,error])

