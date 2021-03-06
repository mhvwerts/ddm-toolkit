# -*- coding: utf-8 -*-
"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

ddm.py:
Numerical engine for calculation of ISF from an arbitrarily long 
sequence of images with minimal memory requirements

    written by Martinus H. V. Werts (CNRS, ENS Rennes, France)
    with contributions from Greshma Babu (IISER Bhopal, India),
    Jai Kumar (IISER Bhopal, India), Nitin Burman (IISER Mohali,
    India)
    

DISTRIBUTED UNDER THE CeCILL LICENSE
https://cecill.info/index.en.html


General remarks
---------------

The present implementation can only handle square input images.

Development of this 'computational engine' benefitted from the 
availabilty of Matlab routines by Helgeson et al.[2] and the tutorial
paper[3] and code[4] on DDM by Germain et al.


References
----------
[1] Cerbino, R.; Trappe, V. 'Differential Dynamic Microscopy: Probing
    Wave Vector Dependent Dynamics with a Microscope'
    Phys. Rev. Lett. 2008, 100, 188102
    https://dx.doi.org/10.1103/PhysRevLett.100.188102

[2] https://sites.engineering.ucsb.edu/~helgeson/ddm.htm

[3] Germain, D.; Leocmach, M.; Gibaud, T. 'Differential Dynamic
    Microscopy to Characterize Brownian Motion and Bacteria Motility'
    Am. J. Phys. 2016, 84, 202–210
    https://dx.doi.org/10.1119/1.4939516

[4] https://github.com/MathieuLeocmach/DDM

[5] Giavazzi, F.; Edera, P.; Lu, P. J.; Cerbino, R. 'Image Windowing
    Mitigates Edge Effects in Differential Dynamic Microscopy'
    Eur. Phys. J. E 2017, 40, 97
    https://dx.doi.org/10.1140/epje/i2017-11587-3


CHANGELOG (for this module)
---------------------------
200424
    - change variable names: qx and qy become ux and uy (to avoid
      confusion with existing literature, factor 2\pi)
    - qx, qy and q re-introduced in the ImageStructureFunction,
      derived from ux and uy: qx = 2*np.pi*ux, qy = 2*np.pi*uy

200213
    - add ISFstep parameter, enabling to calculate ISF only at specific
      lag times, while still using all available data to calculate ISF
      at those points. This is better than using 'skip' which simply
      throws away intermediate frames

200122
    - integrated this module as 'ddm' into 'ddm_toolkit'
    - code and syntax clean-up (unfinished)
    - new functional, intermediate version requiring further
      maintenance in near future!   

190716
    - added Blackman-Harris apodization as suggested by 
      Giavazzi et al.[5]

190206
    - further clean-up
    - include pre-averager/image picker for multitau
    
190130 
    - back to basics, restructure code
    - separated out and removed analysis functions (analysis should be
      done afterwards, elsewhere)

180730 
    - initial working DDM code
"""



import numpy as np

BUFTYPE = np.float

class ImageStructureEngine:
    """Sequential, cumulative image structure function (ISF) engine.
    
    This engine processes an arbitrarily long sequence of images, using
    the minimal amount of memory for calculating the ISF, averaging
    the ISF over many images, if the sequence length permits.
    
    The entire calculation is carried out in pixel-frame units, i.e.
    time lags are in numbers of frames, the Fourier transforms give
    'inverse pixel' frequencies. Book-keeping concerning the conversion
    to real world units (viz. µm/pixel and s/frame) should be done
    elsewhere.
    
    Instantiation of the class prepares a ImageStructureEngine for 
    receiving and processing an image sequence. Several of these engines
    can be created 'in parallel', having different processing options.

    Instantiation
    -------------
    ise = ImageStructureEngine(
        Npx, Nbuf,  ISFstep = 1, apodization = 'No',
        pick = 1, avg = 1, drop = 0, fillbuf = True,
        dummyrun = False
        )
        
    Parameters
    ----------
    Npx : int
        Number of pixel of each frame of input; must be a positive
        multiple of 2 
    Nbuf : int
        Buffer size, number of time lags of the ISF that will be
        calculated; any positive non-zero integer
    ISFstep : int: step-size that defines which lag times of the ISF in
        the buffer will be calculated. Default = 1 which means that all
        lag times will be calculated. ISFstep = 10 will only calculate
        the ISF at lags 0, 10, 20... Nbuf, the other ISF lags will be
        zero. This is different from the 'pick' parameter (see 'Low-level'
        parameters below), since the frames actually do fill up the FIFO
        frame buffer, and contribute to the calculation of the ISF at 
        some point in the sequence. This gives better ISFs at long times,
        at the expense of memory usage, but does avoid long calculation
        times by only calculating some time lags.
        #TODO propose a logarithmic scale step-size! => we should make
        a new parameter that is just a list of the taufs for which the ISF
        is actually calculated. This will also change the structure of
        ISFaccum.
    apodization : str : 'No' or 'Blackman-Harris'
        If not 'No', a 2D Blackman-Harris window will be applied to
        the incoming image before further processing
        
    Low-level tweak parameters
    --------------------------
    pick : int (optional)
        Pick only every Npick-th image for processing into the FIFO 
        image buffer. The lag times will then be spaced by Npick. The
        ISF will cover lags 0 until Npick*Nbuf
    avg : int (optional)
        Instead of picking one image, take the average over Navg
        subsequent images. Allowed values: 1...Npick. 
    drop : int (optional)
        Before actually starting the engine, drop Ndrop images. This is
        intended for centering the picking by one engine around the
        averaging-picking by another engine.
    fillbuf : bool (future parameter, not fully implemented)
        Normally (fillbuf == True) the ISF calculation buffer will
        first be filled BEFORE any ISF is calculated. This way, all lag
        times of the ISF will be calculated over the same number of 
        image pairs. NOT IMPLEMENTED YET: fillbuf==False, in which case 
        ISFs would be calculated for those lags that are already in the
        buffer.
        
    Attributes
    ----------
    tau : int
        Lag times of the ISF (in frame units).
    ux, uy : float
        Spatial frequency coordinates in x and y direction of the 2D FFT.
        These are in 'inverse pixel' units. They are not the
        "radial pixel frequency" q
            q = 2 * np.pi * u
        Spatial sample distance is always 1.0 pixel
    ISFcount : int
        Number of frames over which the total ISF has been
        accumulated. (This is a single int value for fillbuf == True,
        but would be a vector for fillbuf == False)
    (to be continued)
        
    Remarks
    -------
    All calculations are done in floating point. (BUFTYPE = np.float; 
    do not change). Incoming images are converted to floating point.
    
    Only averaging is implemented, since summing of images was found
    to be not useful. The averages are calculated in floating point.
    """    
    def __init__(
            self, Npx, Nbuf,  ISFstep = 1, apodization = 'No',
            pick = 1, avg = 1, drop = 0, fillbuf = True,
            dummyrun = False
            ):
        assert type(Npx) == int, 'Npx should be an integer'
        assert Npx%2 == 0, 'Npx should be a multiple of 2'
        self.Npx = Npx
        assert type(Nbuf) == int, 'Nbuf should be an integer'
        assert Nbuf > 0, 'Nbuf should be positive'
        self.Nbuf = Nbuf
        self.ISFstep = ISFstep
        self.framebuf = np.zeros((Nbuf,Npx,Npx), dtype = BUFTYPE)
        self.frameptr = np.arange(Nbuf) # pointers for LIFO operation 
        #                                 of buffer
        self.framesum = np.zeros((Npx,Npx), dtype = BUFTYPE)
        self.bufN = 0
        self.ISFaccum = np.zeros((Nbuf+1,Npx,Npx)) # Nbuf+1: one extra
        #                                            point to incorporate 
        #                                            delta t = 0
        self.ISFcount = 0 # ISF accumulation counter
        self.totalframes = 0 # total frames processed
        
        if apodization not in ['No', 'Blackman-Harris']:
            raise TypeError('Unknown apodization option (case-sensitive)')
        self.apodwindow = None
        if apodization=='Blackman-Harris':
            a0 = 0.3635819; a1 = 0.4891775
            a2 = 0.1365995; a3 = 0.0106411
            xco = np.linspace(0, 1.0, Npx, dtype = BUFTYPE)
            yco = np.linspace(0, 1.0, Npx, dtype = BUFTYPE)
            X,Y = np.meshgrid(xco,yco)
            Wbhx = a0 -  a1*np.cos(2*np.pi*X) + a2*np.cos(2*np.pi*X*2)\
                   -  a3*np.cos(2*np.pi*X*3)
            Wbhy = a0 -  a1*np.cos(2*np.pi*Y) + a2*np.cos(2*np.pi*Y*2)\
                   -  a3*np.cos(2*np.pi*Y*3)
            self.apodwindow = Wbhx*Wbhy
        
        assert type(avg) == int, 'avg should be int'
        assert avg > 0, 'avg should be > 0'
        assert type(pick) == int, 'pick should be int'
        assert pick >= avg, 'pick should be >= avg'
        self.Npick = pick
        self.Navg = avg
        assert type(drop) == int, 'drop should be int'
        assert drop >= 0, 'drop should be >= 0'
        self.Ndrop = drop
        
        assert fillbuf, 'fillbuf=False not implemented yet'
        
        # dummyrun flag (for testing purposes)
        self.dummyrun = dummyrun
        
        # create axis coordinate vectors       
        self.accumtauf = np.arange(0, self.Nbuf + 1) * self.Npick 
        #   tauf of the ISFaccum buffer (TODO: change ISFaccum buffer
        #   to only hold tau frames that are actually calculated)
        self.tauf = self.accumtauf[::self.ISFstep]
        self.ux = np.fft.fftshift(np.fft.fftfreq(Npx))
        self.uy = np.fft.fftshift(np.fft.fftfreq(Npx))
        
        # set private, internal bookkeeping variables
        self.idrop = 0 # number of dropped frames
        self.ipick = self.Npick - 1 # pick counter
        #      ipick should be initialized such that the
        #      first frame of the sequence (after dropped frames)
        #      triggers the 'pick' flag (see self.push method)
        self.iavg = 0 # averager counter
        self.isum = 0 # count actual frames in average
        
        self.bufdtype = BUFTYPE

    def _ISF(self, img_t, img_t_dt):
        """calculate the Image Structure Function between
        img_t_dt (t + delta_t) and img_t (t)
        """
        if self.dummyrun:
            return np.ones_like(img_t)
        else:
            DI= img_t_dt - img_t
            return np.abs(np.fft.fft2(DI))**2
    
    def _push(self, binframe):
        """directly push a new frame onto the buffer. When the buffer is 
        full, first calculate the ISF between inframe and all images in
        buffer. Then, add this to cumulative ISFaccum
        """
        assert binframe.dtype == BUFTYPE, 'binframe should be of dtype BUFTYPE'
        if self.bufN < self.Nbuf:
            self.framebuf[self.bufN] = binframe
            self.bufN += 1
        else:
            frame_t_dt = binframe
            for itau,pt in enumerate(self.frameptr[-1::-1]):
                if ((itau+1) % self.ISFstep == 0):
                    frame_t = self.framebuf[pt,:,:]
                    self.ISFaccum[itau+1,:,:] +=\
                                        self._ISF(frame_t, frame_t_dt)
            self.ISFcount+=1 # update ISF accumulation counter
            self.frameptr = np.roll(self.frameptr, -1) 
            self.framebuf[self.frameptr[-1]] = binframe

    def push(self, inframe):
        """push a frame onto the buffer after preprocessing
        (picking, averaging, dropping, apodization)
        """
        self.totalframes += 1

        #TODO: the following should (perhaps) be reimplemented using
        #  a more finite-state-machine style (2 phases per pushed frame)
        assert self.ipick < self.Npick, 'There is something horribly wrong! '\
                                        '(Really!!)'
        if (self.idrop < self.Ndrop):
            # drop frames before starting operation
            self.idrop += 1
        else:
            self.ipick += 1
            pick_flag = (self.ipick == self.Npick)
            if pick_flag:
                self.ipick = 0
                self.iavg = 0
                self.framesum.fill(0.0)
                self.isum = 0
            if (self.iavg < self.Navg):
                binframe = inframe.astype(BUFTYPE)  
                self.framesum += binframe
                self.isum += 1
            self.iavg += 1
            if (self.iavg == self.Navg):
                pushframe = self.framesum/self.Navg
                if self.apodwindow is None:
                    self._push(pushframe)
                else:
                    self._push(pushframe*self.apodwindow)

    def reset(self):
        """reset the ISFengine to its freshly initialized state"""
        self.ISFaccum[:,:,:] = 0
        self.ISFcount = 0 
        self.bufN = 0
        self.ISFcount = 0 
        self.totalframes = 0 
        self.idrop = 0 
        self.ipick = self.Npick - 1 # pick counter
        self.iavg = 0 
        self.isum = 0 
        #TODO re-verify that these resets are coherent with __init__
        #compare to what happens in __init__, which by the way
        #should be cleaned-up and organized (first set the attributes)
        #then set the internal counters
    
    def ISFframe(self, itau):
        """output ISF frame (2D image) at time index itau
        this function performs fft shifting such that the output can be 
        easily shown using imshow, and processed"""
        return np.fft.fftshift(self.ISFaccum[itau])/self.ISFcount

    def ISF(self):
        """return a new matrix with the full, correctly oriented
        ('fftshift'-ed) ISF[t,y,x]"""
        assert self.ISFcount > 0, 'No ISF is available (not enough '\
                                  'image frames to fill buffer?)'
        Nframes = self.ISFaccum.shape[0]
        fullISF = np.zeros_like(self.ISFaccum[::self.ISFstep])
        for frmi,accumfrmi in enumerate(range(0, Nframes, self.ISFstep)):
            fullISF[frmi,:,:] = self.ISFframe(accumfrmi)
        return fullISF
 
    def save(self, fpn):
        """save the ISF to an NPZ file, including the ISFengine settings.
        
        This will not include the framebuffer, since it takes up quite 
        some memory and is not useful for further analysis. This 'save'
        method is intended to save the result of ISF calculation, rather
        than storing the complete state of the ISFengine object.        
        
        The accumulated ISF in ISFaccum will be 'fftshifted' into 
        displayable and easily processable form. The output will be 
        written to a file 'outputfile.npz' which contains Npx, Nbuf, 
        ISFcount, totalframes, preNf, premode, ISF.
        ISFcount = Total no. of ISF in file. 
        totalframes = Total no. of frames processed.
        ISF = 3D array containing 2D Image Structure at all time lag
        .
        """
        hdic ={}
        hdic['Npx'] = self.Npx
        hdic['Nbuf'] = self.Nbuf
        hdic['ISFcount'] = self.ISFcount
        hdic['totalframes'] = self.totalframes
        hdic['Npick'] = self.Npick
        hdic['Navg'] = self.Navg
        hdic['Ndrop'] = self.Ndrop
        hdic['apodwindow'] = self.apodwindow
        fullISF = self.ISF()
        np.savez(fpn,
                 header = hdic,
                 ISF = fullISF, 
                 tauf = self.tauf, uy = self.uy, ux = self.ux
                 )

         
        
class ImageStructureFunction:
    """Class containing an ISF in the form of an array
    together with scaling information
    
    Can be constructed by loading a suitable NPZ file (output of
    ImageStructureEngine.save)
    
        isf = ImageStructureFunction.fromfilename(NPZfilename)
    
    The class can furthermore be instantiated directly from an
    ImageStructureEngine (no need to save intermediate ISF result)
    
        isf = ImageStructureFunction.fromImageStructureEngine(ISE_object)
    
    Additional method provided by this class:
        
    radavg(itau) = It will give radial average of ISF at a 
    particular lag time index. The corresponding lag time is given by
    tau[itau]
    """
    def __init__(self, ISFarray, tauf, ux, uy, hdic=None):
        self.ISF = ISFarray
        self.tauf = tauf
        self.ux = ux
        self.uy = uy
 
        assert len(self.ux) == len(self.uy), "only square ISF accepted"       
        self.Npx = len(self.ux)
       
        # actually not sure if the following is useful
        # probably better keep program to minimum
        # perhaps we can just store the header as self.hdic
        # instead of decoding here
        if hdic is not None: #TODO: check if we can analyze ISF without
            #                 having access to these data
            # self.Npx = hdic['Npx'] #perhaps not needed!
            self.Nbuf = hdic['Nbuf']
            self.ISFcount = hdic['ISFcount']
            self.totalframes = hdic['totalframes']
            self.Npick = hdic['Npick']
            self.Navg = hdic['Navg']
            self.Ndrop = hdic['Ndrop']
            self.apodwindow = hdic['apodwindow']
        
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
        
        #initialize this to pixel, frame units
        # replace with 'real world' units by calling 
        #   real_world(um_p_px, s_p_frm)
        self.real_world_units = False
        self.q = 2*np.pi*self.u
        self.qx = 2*np.pi*self.ux
        self.qy = 2*np.pi*self.uy
        self.tau = self.tauf * 1.0

    @classmethod
    def fromfilename(cls, fpn):
        with np.load(fpn, allow_pickle=True) as npz:
            hdic = npz['header'].item()
            tauf = npz['tauf']
            uy = npz['uy']
            ux = npz['ux']
            ISF = npz['ISF']
        return cls(ISF, tauf, ux, uy, hdic)

    @classmethod
    def fromImageStructureEngine(cls, ISE_instance):
        #TODO: copy rest of attributes into dictionary
        # for now, do not use dictionary, in order to test independence
        return cls(ISE_instance.ISF(), ISE_instance.tauf,
                   ISE_instance.ux, ISE_instance.uy)

    def real_world(self, um_p_px, s_p_frm):
        """scale q wavevectors and tau time lags to real-world units
        
        q, qx, qy are initially initialized to pixels^-1 (radians)
        tau is initially initialized to frames
        
        By calling real_world, these are scaled to real-world units,
        in our case µm - seconds, to keep it simple. Of course, this
        choice is arbitrary.
        
        To indicate that real world units have been set, the flag
            real_world_units is set to True
        """
        
        self.q = 2*np.pi*self.u / um_p_px
        self.qx = 2*np.pi*self.ux / um_p_px
        self.qy = 2*np.pi*self.uy / um_p_px
        self.tau = self.tauf * s_p_frm
        self.real_world_units = True

        
    def radavg(self, itau):
        """Extract radial average of ISF at index itau.
        The corresponding time lag is self.tau[itau].
        Correspond lag in number of frames is self.tauf[itau]
        the ordinates are in qrs"""
        sumbinpix = np.histogram(self.dists, self.bins,
                            weights = self.ISF[itau])[0]
        return sumbinpix/self.Nbinpix
    

    
        
