"""
A new, minimalist implementation of the DDM computational machinery

M. H. V. Werts
MOLTECH-Anjou, CNRS, Université d'Angers
"""
import numpy as np
from scipy import fft



class DifferentialImageCorrelator:
    def __init__(self, Npx, Nbuf, tauix):
        """
        Computational engine for calculation of DICF from image stream
        
        The DICF is also known as the (image) structure function.
        
        Parameters
        ----------
        Npix : int
            Width and length of square image frame in number of pixels.
        Nbuf : int
            Number of frames in buffer.
        tauix : list of int, np.ndarray(dtype=int)
            Indices of tau for which the DICF is evaluated. It is easy to 
            create an arbitrary time scale, for example a linear or a 
            logarithmic time-scale
        """
        
        # just make sure, unexpected things may occur if not int
        assert type(Npx) is int
        assert type(Nbuf) is int
        # TODO also check tauix? but this is more cumbersome 
        
        # Engine parameters
        self.tauix = tauix
        self.Ntau = len(self.tauix)
        self.Npx = Npx
        self.Nbuf = Nbuf
        self.ux = fft.rfftfreq(self.Npx)
        self.uy = fft.fftshift(fft.fftfreq(self.Npx))
        
        # Engine state
        # FFT buffer (for RFFT2)
        self.fftbuf_real = np.zeros((Nbuf, Npx, Npx//2+1), dtype = np.float64)
        self.fftbuf_imag = np.zeros((Nbuf, Npx, Npx//2+1), dtype = np.float64)
        # DICF accumulator
        self.DICF_accum = np.zeros((self.Ntau, Npx, Npx//2+1), dtype = np.float64)       
        # index of last written circular buffer memory slot
        self.fftbuf_ix_in = 0 
        # number of frames pushed to buffer
        self.totalframes = 0
        # number of frames contributing to DICF (for normalization)
        self.DICF_Nframes = 0
        
    
    @classmethod
    def log_tau(cls, Npx, Nbuf, Ntau_req):
        """
        Instantiate a DICF engine for calculation of DICF, with logarithmic 
        time scale


        Parameters
        ----------
        Npx : int
            Width and length of square image frame in number of pixels.
        Nbuf : int
            Number of frames in buffer.
        Ntau_req : int
            Requested number of values of tau for which the DICF is evaluated. 
            A logarithmic time scale will be created. The actual number of 
            tau points will always be lower, because of several subsequent 
            calculated values being rounded to the same integer index.


        Returns
        -------
        Instance of DifferentialImageCorrelator.

        """
        return cls(Npx, Nbuf, np.unique(
                                 np.array(
                                   np.round(np.logspace(0, 
                                                        np.log10(Nbuf), 
                                                        Ntau_req,
                                                        endpoint=True)),
                                   dtype = int)))
    
    @classmethod
    def lin_tau(cls, Npx, Nbuf):
        """
        Instantiate a DICF engine for calculation of DICF, with a
        standard linear time scale containing Nbuf points

        Parameters
        ----------
        Npx : int
            Width and length of square image frame in number of pixels.
        Nbuf : int
            Number of frames in buffer.
        

        Returns
        -------
        Instance of DifferentialImageCorrelator.

        """
        return cls(Npx, Nbuf, np.arange(1, Nbuf+1))
        
    
    def push(self, img_in):
        # process incoming frame
        # TODO: optimize threads/workers for rfft2?
        img_in_fft = fft.rfft2(img_in)
        
        # only when all (circular) buffer positions have been filled
        if (self.totalframes >= self.Nbuf):
            # img_in is frame "t + dt"
            # frame in buffer is frame "t"
            self._update_DICF(img_in_fft.real, img_in_fft.imag)
            self.DICF_Nframes += 1

        # update index for writing incoming frame (RFFT2) to circular buffer
        if (self.fftbuf_ix_in <= 0):
            self.fftbuf_ix_in = self.Nbuf
        self.fftbuf_ix_in -= 1

        # Write fft  of incoming image to buffer slot ix_in
        # Separating the real and imaginary parts here leads to a >10%
        # overall speed increase, thanks to more efficient _update_DICF loop
        self.fftbuf_real[self.fftbuf_ix_in, :, :] = img_in_fft.real
        self.fftbuf_imag[self.fftbuf_ix_in, :, :] = img_in_fft.imag

        # frame counter
        self.totalframes += 1
        
        
    def _update_DICF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        No checks are done on img_in: we trust that they are np.array of 
        shape (Npx,Npx) with dtype=np.float64
        
        TODO: Accelerate using Numba?
        """
        for n in range(self.Ntau):
            ixc = (self.tauix[n] + self.fftbuf_ix_in - 1) % self.Nbuf
            dDICF = (img_in_fft_real - self.fftbuf_real[ixc,:,:])**2 \
                  + (img_in_fft_imag - self.fftbuf_imag[ixc,:,:])**2
            # add to total DICF
            self.DICF_accum[n,:,:] += dDICF
            
    
    def get_DICF(self):
        """return a new matrix with the full, correctly oriented
        ('fftshift'-ed) DICF[tau,y,x]
        
        Returns the full, square DICF.
        
        TODO: option to return only half (RFFT2-based) DICF?
        """
        assert self.DICF_Nframes > 0, 'No DICF is available (not enough '\
                                  'image frames to fill buffer?)'
            
        DICF_arr = np.zeros((self.Ntau, self.Npx, self.Npx)) 
        for ix, dDICF in enumerate(self.DICF_accum):
            # Reconstruct full, square DICF frame
            #  using the symmetry of the FFT2 vs RFFT2.
            # The correctness of the reconstruction is explicitly verified 
            #  in test_5, by comparing the full DICF ('ISF') obtained through
            # this (RFFT2) algorithm to the 'classic' (FFT2-based ) algorithm.
            halfDICF = fft.fftshift(dDICF, axes=[0])/self.DICF_Nframes
            DICF_arr[ix, 0, 0:self.Npx//2] = halfDICF[0, -1:0:-1]
            DICF_arr[ix, 1:, 0:self.Npx//2] = halfDICF[-1:0:-1, -1:0:-1]
            DICF_arr[ix, :, self.Npx//2:] = halfDICF[:, 0:-1]
            
        # For consistency check, do an 'official' reconstruction also for
        # coordinates
        # DICF_ux = np.concatenate((-self.ux[self.Npx//2:0:-1],
        #                           self.ux[:-1]))
        DICF_ux = self.uy # this is quicker
        DICF_uy = self.uy
        return DICF_arr, DICF_uy, DICF_ux
        
        

class DifferentialImageCorrelationFunction:
    """Class containing a DICF in the form of an array
    together with scaling information
    
    The DICF is also known as the (image) structure function.

    The class can be instantiated directly from a
    DifferentialImageCorrelator object

        dicf = DifferentialImageCorrelationFunction\
               .fromDifferentialImageCorrelator\
               (DifferentialImageCorrelator_instance)

    Can be constructed by loading a suitable NPZ file (output of
    DifferentialImageCorrelationFunction.to_file)

        dicf = DifferentialImageCorrelationFunction\
               .from_file(NPZfilename)



    Additional method provided by this class:

    radavg(itau) = It will give radial average of DICS at a
    particular lag time index. The corresponding lag time is given by
    tau[itau]
    """
    def __init__(self, dicf_array, tauix, uy, ux):
        self.values = dicf_array
        self.tauix = tauix
        self.Ntau = len(self.tauix)
        self.ux = ux
        self.uy = uy

        # Just checking...
        assert len(self.ux) == len(self.uy), "only square DICF accepted"
        assert self.values.shape[1] == len(self.ux), "inconsistent DICF array shape"
        assert self.values.shape[2] == len(self.uy), "inconsistent DICF array shape"
        self.Npx = len(self.uy) # original dimensions
        assert self.values.shape[0] == self.Ntau, "inconsistent DICF array shape"
        
        # initialize radial averager
        self.dists = np.sqrt(self.uy[:,None]**2 +
                             self.ux[None,:]**2)
        self.dists[self.Npx//2,:] = 0. # central cross goes in q = 0 bin
        self.dists[:,self.Npx//2] = 0.
        ux0 = self.ux[self.Npx//2]
        assert ux0 == 0.0
        ux1 = self.ux[self.Npx//2+1]
        halfstep = (ux1 - ux0)/2.0
        assert halfstep > 0
        self.bins = np.append(-halfstep, self.ux[self.Npx//2:]+halfstep)
        histo = np.histogram(self.dists, self.bins)
        self.Nbinpix = histo[0]
        self.u = (histo[1][0:-1]+histo[1][1:])/2.0

        # Initialize this to pixel, frame units, indicating that these
        # are not 'real world' units
        #
        # Will be replaced with valid 'real world' units by calling
        #   real_world(um_p_px, s_p_frm)
        self.real_world_units = False
        self.q = 2*np.pi*self.u
        self.qx = 2*np.pi*self.ux
        self.qy = 2*np.pi*self.uy
        self.tau = self.tauix * 1.0
        
        
    @classmethod
    def fromDifferentialImageCorrelator(cls, dimgcorr):
        DICFarray, uy, ux = dimgcorr.get_DICF()
        return cls(DICFarray, dimgcorr.tauix,
                   uy, ux)


    @classmethod
    def from_file(cls, fpn):
        with np.load(fpn, allow_pickle=True) as npz:
            tauix = npz['tauix']
            uy = npz['uy']
            ux = npz['ux']
            DICFarray = npz['DICFarray']
            DICF1 = cls(DICFarray, tauix, uy, ux)
            DICF1.q = npz['q']
            DICF1.qx = npz['qx']
            DICF1.qy = npz['qy']
            DICF1.tau = npz['tau']
            DICF1.um_p_px = npz['um_p_px']
            DICF1.s_p_frm = npz['s_p_frm']
        return DICF1


    def to_file(self, fpn):
        """
        Save the ImageStructureFunction to a file.

        This generates a file that is identical to (or at least compatible
        with) a file generated using ImageStructureEngine.saveISF method.

        TODO: USE HDF5/NETCDF4

        Parameters
        ----------
        fpn : str
            File pathname.

        Returns
        -------
        None.

        """
        assert self.real_world_units, "apply 'real world' units before saving."
        np.savez(fpn,
                 DICFarray = self.values,
                 tauix = self.tauix, uy = self.uy, ux = self.ux,
                 q = self.q, qx = self.qx, qy = self.qy,
                 tau = self.tau,
                 um_p_px = self.um_p_px, 
                 s_p_frm = self.s_p_frame,
                 )


    def real_world(self, um_p_px, s_p_frm):
        """Scale wavevectors (q) and time lags (tau) to real-world units
        
        Parameters
        ----------
        um_p_px : float
            micrometers per pixel (square pixels).
        s_p_frm : float
            seconds per frame (inverse of frame rate).

        Returns
        -------
        None.

        q, qx, qy are initially initialized to pixels^-1 (radians)
        tau is initially initialized to frames

        By calling real_world, these are scaled to real-world units,
        in our case µm and seconds.

        To indicate that real world units have been set, the flag
            real_world_units is set to True
        """
        self.um_p_px = um_p_px
        self.s_p_frame = s_p_frm
        self.q = 2*np.pi*self.u / um_p_px
        self.qx = 2*np.pi*self.ux / um_p_px
        self.qy = 2*np.pi*self.uy / um_p_px
        self.tau = self.tauix * s_p_frm
        self.real_world_units = True


    def radavg(self, itau):
        """Extract radial average of DICF at index itau.
        
        The corresponding time lag is self.tau[itau].
        Correspond lag in number of frames is self.tauix[itau]
        the ordinates are in qrs"""
        sumbinpix = np.histogram(self.dists, self.bins,
                            weights = self.values[itau])[0]
        return sumbinpix/self.Nbinpix

        

