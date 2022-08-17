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

210708
    - added three new ImageStructureEngine models, of which two use
      Numba-based acceleration (running on CPU)
    - The mechanics of choosing and instantiating a particular model of
      ImageStructureEngine was simplified (at least for the high-level user).
      From now on there is just one "ImageStructureEngine" factory, that
      generates an instance of a particular ImageStructureEngine model,
      selected via a 'engine_model' keyword parameter. All other parameters
      work the same as previously, and the change is transparent for
      high-level usage of the toolkit.

210208
    - implemented two alternative ImageStructureEngine classes.

      ImageStructureEngine2 puts all FFTs to be calculated into a single
      stack, and then does all FFTs using a single call. This will facilitate
      interfacing with CUDA or FFTW.
      This alternative engine seemed promising for simple CUDA/FFTW speed up.
      However, reducing the number of necessary FFTs is a more efficient way
      of optimizing the code (see ImageStructureEngine3)

      ImageStructureEngine3 greatly reduces the number of individual
      FFTs to be calculated, by using linearity of FFT and storing FFTs already
      calculated. "Only" 50% perfomance boost observed; the subsequent complex
      arithmetic (substracting, abs value) is probably heavy as well.
      We imagine that the algorithm might run entirely on
      a GPU, with all buffering and operations taking place directly on GPU.

200424
    - change variable names: qx and qy become ux and uy (to avoid
      confusion with existing literature, factor 2 pi)
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

#from matplotlib import pyplot as plt


# Numba/CPU dependent functions
# If Numba cannot be imported, set 'numba' to 'None'.
# The fact that 'numba' is 'None' when Numba not available can be used
# to generate error message in case the user calls for Numba-accelerated
# functionality.
try:
    import numba
except:
    numba = None

if not (numba is None):
    from .ddm_numba_cpu import numba_ISF_push_frame_fft_v1
    from .ddm_numba_cpu import numba_ISF_push_frame_fft_v2
    from .ddm_numba_cpu import numba_ISF_push_frame_fft_v2_corr_LB


# Numba/CUDA (GPU) dependent functions
# Conditional imports giving 'cuda is None' if CUDA GPUs not available.
try:
    import numba.cuda as cuda
except:
    cuda = None

if not (cuda is None):
    if not cuda.is_available():
        cuda = None
    else:
        print()
        print('CUDA GPU detected...')
        cuda.detect() # this function also returns True if GPU available
        print()
        from .ddm_numba_cuda import ImageStructureEngine7


# Enabling (potential) performance tweaking
BUFTYPE = np.float32       # might be tuned later for performance
                           # np.float32 is ~25% faster with original ISFEngine
                           #   on my computer

FFTBUFTYPE = np.complex64  # might be tuned later for performance
                           #    did not find speed difference between complex64
                           #    and complex128 on my compyter


#

available_engine_models = [1,2,3,4]

best_available_engine_model = 4 # best numpy-only model
if not (numba is None):
    available_engine_models.append(5)
    available_engine_models.append(6)
    best_available_engine_model = 6 # with numba available
if not (cuda is None):
    available_engine_models.append(7)
    best_available_engine_model = 7


#%%
# "First-generation" ImageStructureEngines:
# ImageStructureEngine models 1, 2, 3
# These three models are based a commmon `ImageStructureEngineBase_1st_gen` class
#


class ImageStructureEngineBase_1st_gen:
    """Base class for ImageStructureEngine models 1, 2, 3

    This class should not be used directly as an ImageStructureEngine,
    but should be sub-classed

    see below:
        ImageStructureEngine
        ImageStructureEngine2
        ImageStructureEngine3

    """
    def __init__(
            self, Npx, Nbuf, apodization = 'No', ISFstep = 1,
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
        self.frameptr = np.arange(Nbuf) # pointers for LIFO operation
        #                                 of buffer
        self.framesum = np.zeros((Npx,Npx), dtype = BUFTYPE)
        self.bufN = 0
        self._ISFaccum = np.zeros((Nbuf+1,Npx,Npx)) # Nbuf+1: one extra
        #                                            point to incorporate
        #                                            delta t = 0
        #                            !!!WARNING!!! In 1st generation engines,
        #                               _ISFaccum contains "itau = 0" as the
        #                           first frame (completely useless, but it
        #                           is there). Just remember, that in
        #                           2nd generation engines, the _ISFaccum
        #                           frames start at "itau = 1"
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
        #   tauf of the _ISFaccum buffer
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

    def _push(self, binframe):
        pass # in this baseclass we might even raise an error when calling this?

    def push(self, inframe):
        """push a frame onto the buffer after preprocessing
        (picking, averaging, dropping, apodization)
        """
        self.totalframes += 1
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
        self._ISFaccum[:,:,:] = 0
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
        return np.fft.fftshift(self._ISFaccum[itau])/self.ISFcount

    def ISF(self):
        """return a new matrix with the full, correctly oriented
        ('fftshift'-ed) ISF[t,y,x]"""
        assert self.ISFcount > 0, 'No ISF is available (not enough '\
                                  'image frames to fill buffer?)'
        Nframes = self._ISFaccum.shape[0]
        fullISF = np.zeros_like(self._ISFaccum[::self.ISFstep])
        for frmi,accumfrmi in enumerate(range(0, Nframes, self.ISFstep)):
            fullISF[frmi,:,:] = self.ISFframe(accumfrmi)
        return fullISF

    def saveISF(self, fpn):
        """
        Save the accumulated Image Structure Functio (ISF) directly to a file.

        This generates a file that is identical to (or at least compatible
        with) a file generated using ImageStructureFunction.save method.
        The file can only be read using the ImageStructureFunction.fromFile
        class method


        Parameters
        ----------
        fpn : str
            File pathname.

        Returns
        -------
        None.


        This method only saves the ISF. It will not include the framebuffer,
        since it takes up quite some memory and is not useful for further
        analysis. This 'saveISF' method is intended to save the result of ISF
        calculation, rather than storing the complete state of the ISFengine
        object.

        The accumulated ISF in _ISFaccum will be 'fftshifted' into
        displayable and easily processable form. The output will be
        written to a file 'outputfile.npz' which contains Npx, Nbuf,
        ISFcount, totalframes, preNf, premode, ISF.
        ISFcount = Total no. of ISF in file.
        totalframes = Total no. of frames processed.
        ISF = 3D array containing 2D Image Structure at all time lag
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



class ImageStructureEngine0(ImageStructureEngineBase_1st_gen):
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

    This is the original version, without any optimizations of the calculations.
    It is the 'tried and tested' reference version.
    Any optimizations should be implemented in other subclasses
    of ImageStructureEngineBase_1st_gen.

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
        _ISFaccum.
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
    All calculations are done in floating point. (BUFTYPE = np.float).
    Incoming images are converted to floating point.

    Only averaging is implemented, since summing of images was found
    to be not useful. The averages are calculated in floating point.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('*** using ImageStructureEngine (the original) ***')
        self.framebuf = np.zeros((self.Nbuf, self.Npx, self.Npx),
                                 dtype = BUFTYPE)

    def _calcISFdt(self, img_t, img_t_dt):
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
        buffer. Then, add this to cumulative _ISFaccum
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
                    self._ISFaccum[itau+1,:,:] +=\
                                        self._calcISFdt(frame_t, frame_t_dt)
            self.ISFcount+=1 # update ISF accumulation counter
            self.frameptr = np.roll(self.frameptr, -1)
            self.framebuf[self.frameptr[-1]] = binframe



class ImageStructureEngine2(ImageStructureEngineBase_1st_gen):
    """Alternative Image Structure Engine based on ImageStructureEngine

    This alternative version will buffer up all FFT calculations such
    that the entire stack of 2D FFTs for each new incoming frame
    can be launched with a single FFT command. This should open the road
    to facile optimization/speed-up of this
    computationally intensive process (use of multiprocessing and/or GPU)

    It also contains an optimized calculation of np.abs(FFT)**2

    The idea for this engine was to combine it later with specific, optimized
    FFT libraries

    GPU -> CuPy
    CPU -> multithreaded FFT
        -> pyFFTW

    FFT2D boosting in Python
        https://www.jparker.me/blog/fft_2d_performance

    pyfftw:
        https://hgomersall.github.io/pyFFTW/pyfftw/interfaces/interfaces.html
        https://github.com/pyFFTW/pyFFTW/issues/264
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('*** using ImageStructureEngine2 ***')
        self.framebuf = np.zeros((self.Nbuf, self.Npx, self.Npx),
                                 dtype = BUFTYPE)
        self.Nfftbuff = self.Nbuf // self.ISFstep
        self.fftbuff = np.zeros((self.Nfftbuff, self.Npx, self.Npx),
                                dtype = BUFTYPE)

    def _push(self, binframe):
        """directly push a new frame onto the buffer. When the buffer is
        full, first calculate the ISF between inframe and all images in
        buffer. Then, add this to cumulative _ISFaccum

        In this alternative version, all difference images are first stored
        in a image processing buffer.
        The FFTs on the entire stack are obtained from a single call of
        fft2.
        """
        assert binframe.dtype == BUFTYPE, 'binframe should be of dtype BUFTYPE'
        if self.bufN < self.Nbuf:
            self.framebuf[self.bufN] = binframe
            self.bufN += 1
        else:
            frame_t_dt = binframe

            # Fill up FFT buffer
            for itau,pt in enumerate(self.frameptr[-1::-1]):
                if ((itau+1) % self.ISFstep == 0):
                    frame_t = self.framebuf[pt,:,:]
                    DI = frame_t_dt - frame_t
                    self.fftbuff[itau//self.ISFstep,:,:] = DI[:,:]

            # DO FFT (on stack of images)
            # fft2 will use the two last dimensions for fft
            # and the first dimension makes the stack
            FFTstack = np.fft.fft2(self.fftbuff)

            ## original:
            # ISF = np.abs(FFTstack)**2
            ## faster:
            ISF = FFTstack.real**2 + FFTstack.imag**2

            # updating _ISFaccum
            for itau,pt in enumerate(self.frameptr[-1::-1]):
                if ((itau+1) % self.ISFstep == 0):
                    self._ISFaccum[itau+1,:,:] +=\
                                        ISF[itau//self.ISFstep,:,:]

            self.ISFcount+=1 # update ISF accumulation counter
            self.frameptr = np.roll(self.frameptr, -1)
            self.framebuf[self.frameptr[-1]] = binframe



class ImageStructureEngine3(ImageStructureEngineBase_1st_gen):
    """Alternative ImageStructureEngine using the linearity of FFT

    FFT(a-b) = FFT(a)-FFT(b)

    which means that results of FFTs may be buffered and recalled

    In practice, this gives 'only' a 50% speed boost in the 'simul' cycle
    (1.5x faster)

    Additionally, we speed up calculation of magnitude squared of FFT difference.
    (2x faster).

    Overall speed increase: 3x compared to original "Model 1" ISEngine,
    while still being 'pure numpy'.

    The ImageStructureEngine3 algorithm is probably similar to the one proposed
    by Cerchiari et al. (Rev. Sci. Instrum. 83, 106101 (2012))
    https://doi.org/10.1063/1.4755747

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('*** using ImageStructureEngine3 ***')
        self.framefftbuf = np.zeros((self.Nbuf, self.Npx, self.Npx),
                                 dtype = FFTBUFTYPE)


    def _push(self, binframe):
        """push the 2D FFT of incoming frame onto the buffer. When the buffer is
        full, calculate the ISF between inframe and all images in
        buffer using the stored FFTs of the images.
        Then, add this to cumulative _ISFaccum
        """
        assert binframe.dtype == BUFTYPE, 'binframe should be of dtype BUFTYPE'

        #TODO: speed up
        binframefft = np.fft.fft2(binframe) # fft may be sped up (FFTW, CUDA)

        if self.bufN < self.Nbuf:
            self.framefftbuf[self.bufN,:,:] = binframefft[:,:]
            self.bufN += 1
        else:
            framefft_t_dt = binframefft

            #TODO: speed up loop below (significant gain is expected here)
            for itau,pt in enumerate(self.frameptr[-1::-1]):
                if ((itau+1) % self.ISFstep == 0):
                    framefft_t = self.framefftbuf[pt,:,:]

                    ## originally:
                    # ISF = np.abs(framefft_t_dt - framefft_t)**2

                    ## first variant: (~2 times faster than original)
                    ISF = (framefft_t_dt.real - framefft_t.real)**2 \
                         + (framefft_t_dt.imag - framefft_t.imag)**2
                    ## numba might make this faster (see docstring)

                    ## second variant: (slightly slower than first var)
                    # dfft = framefft_t_dt - framefft_t
                    # ISF = dfft.real**2 + dfft.imag**2

                    self._ISFaccum[itau+1,:,:] += ISF


            self.ISFcount+=1 # update ISF accumulation counter
            self.frameptr = np.roll(self.frameptr, -1)
            self.framefftbuf[self.frameptr[-1],:,:] = binframefft[:,:]



#%%
# "Second-generation" ImageStructureEngines


class ImageStructureEngine4:
    """A streamlined ImageStructureEngine (pure numpy)
    
    This engine is computationally identical to Model#3
    but has simpler code
    
    This model is the basis for Numba CPU and Numba CUDA
    optimizations
    """
    def __init__(self, Npx, Nbuf, apodization = None):
        # get parameters: image size (Npx times Npx) and ISF depth (Nbuf)
        self.Npx = Npx
        self.Nbuf = Nbuf
        # define buffer memory (stores FFT results)
        self.BUFreal = np.zeros((Nbuf, Npx, Npx), dtype = np.float64)
        self.BUFimag = np.zeros((Nbuf, Npx, Npx), dtype = np.float64)
        # index of last written circular buffer memory slot
        self.ix_in = 0 
        # number of frames pushed to buffer
        self.totalframes = 0
        # number of frames contributing to ISF (for normalization)
        self.ISFcount = 0
        # ISF: image structure function array (cumulative)
        self._ISFaccum = np.zeros((Nbuf, Npx, Npx))
        
        # set frame/pixel scaling
        self.tauf = np.arange(self.Nbuf+1) * 1.0
        self.ux = np.fft.fftshift(np.fft.fftfreq(self.Npx))
        self.uy = np.fft.fftshift(np.fft.fftfreq(self.Npx))
        
        # the following properties have no use in this engine
        # they are (for now) need for compatibility with first-generation code
        self.Npick = 1
        self.Navg = 1
        self.Ndrop = 0
        
        #TODO apodization to be implemented
        self.apodwindow = None
        if apodization is not None:
            print('WARNING: Apodization not yet implemented on ImageStructureEngine4. No apodization applied.')
        
    def push(self, img_in):
        """Push frame to Image Structure Engine

        This outside function handles the circular (FIFO) buffer.
        It then calls the image structure calculation function

        """
        # TODO: incoming frame: apply apodization (Blackman-Harris window etc.)
        
        # process incoming frame
        img_in_fft = np.fft.fft2(img_in)
        
        # only when all (circular) buffer positions have been filled
        if (self.totalframes >= self.Nbuf):
            # img_in is frame "t + dt"
            # frame in buffer is frame "t"
            self._update_ISF(img_in_fft.real, img_in_fft.imag)
            self.ISFcount += 1

        # update index for writing incoming frame (FFT2) to circular buffer
        if (self.ix_in <= 0):
            self.ix_in = self.Nbuf
        self.ix_in -= 1

        # write fft  of incoming image to buffer slot ix_in
        # GPU: this should be done using buffers in GPU memory
        self.BUFreal[self.ix_in, :, :] = img_in_fft.real
        self.BUFimag[self.ix_in, :, :] = img_in_fft.imag

        # frame counter
        self.totalframes += 1

    def _update_ISF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """
        for n in range(self.Nbuf):
            ixc = (n + self.ix_in) % self.Nbuf
            dISF = (img_in_fft_real - self.BUFreal[ixc,:,:])**2 \
                  + (img_in_fft_imag - self.BUFimag[ixc,:,:])**2
            # add to total isf
            self._ISFaccum[n,:,:] += dISF
                
    def ISF(self):
        """return a new matrix with the full, correctly oriented
        ('fftshift'-ed) ISF[t,y,x]"""
        assert self.ISFcount > 0, 'No ISF is available (not enough '\
                                  'image frames to fill buffer?)'
            
        ISFarray = np.zeros((self.Nbuf+1,self.Npx,self.Npx)) 
        # in ImageStructureFunction ISFarray, the array is offset by 1
        # this is because ISF[0] contains delta t = 0
        for ix, dISF in enumerate(self._ISFaccum):
            ISFarray[ix+1,:,:] = np.fft.fftshift(dISF)/self.ISFcount
    
        return ISFarray
   
      
   
class ImageStructureEngine5(ImageStructureEngine4):
    """Single-threaded numba-version of ImageStructureEngine4
    """
    def _update_ISF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """
        # call numba
        numba_ISF_push_frame_fft_v1(img_in_fft_real, img_in_fft_imag,
                                    self._ISFaccum, self.ix_in,
                                    self.BUFreal, self.BUFimag,
                                    self.Nbuf)



class ImageStructureEngine6(ImageStructureEngine4):
    """Multi-threaded numba-version of ImageStructureEngine4
    """
    def _update_ISF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """
        # call numba
        numba_ISF_push_frame_fft_v2(img_in_fft_real, img_in_fft_imag,
                                    self._ISFaccum, self.ix_in,
                                    self.BUFreal, self.BUFimag,
                                    self.Nbuf)






#%% New ImageStructureEngines by Lancelot Barthe
#
# These are available as number 14 (LB4), number 15 (LB5), number 16 (LB6)

class ImageStructureEngineLB4:
    """A streamlined ImageStructureEngine (pure numpy)
    Tuned by Lancelot Barthe 

    This engine is computationally identical to Model#3
    but has simpler code

    This model is the basis for Numba CPU and Numba CUDA
    optimizations
    """
    def __init__(self, Npx, Nbuf,apodization = 'No', Nacc = None):
        # get parameters: image size (Npx times Npx) and ISF depth (Nbuf) and ISF accumulator size
        self.Npx = Npx
        self.Nbuf = Nbuf
        self.Nacc = Nacc
        # define buffer memory (stores FFT results)
        self.BUFreal = np.zeros((Nbuf, Npx, Npx), dtype = np.float64)
        self.BUFimag = np.zeros((Nbuf, Npx, Npx), dtype = np.float64)
        # index of last written circular buffer memory slot
        self.ix_in = 0
        # number of frames pushed to buffer
        self.totalframes = 0
        # number of frames contributing to ISF (for normalization)
        self.ISFcount = 0
        # ISF: image structure function array (cumulative)
        self._ISFaccum = np.zeros((Nbuf, Npx, Npx))
        #self._ISFaccum_list = [np.zeros((Nbuf, Npx, Npx))]*
        #%
        self.apodization = apodization

        if self.apodization not in ['No', 'Blackman-Harris']:
            raise TypeError('Unknown apodization option (case-sensitive)')
        self.apodwindow = None
        if self.apodization=='Blackman-Harris':
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

        # set frame/pixel scaling
        self.tauf = np.arange(self.Nbuf+1) * 1.0
        self.ux = np.fft.fftshift(np.fft.fftfreq(self.Npx))
        self.uy = np.fft.fftshift(np.fft.fftfreq(self.Npx))

        # the following properties have no use in this engine
        # they are (for now) need for compatibility with first-generation code
        self.Npick = 1
        self.Navg = 1
        self.Ndrop = 0

    def push(self, img_in):
        """Push frame to Image Structure Engine

        This outside function handles the circular (FIFO) buffer.
        It then calls the image structure calculation function

        """

        # process incoming frame
        if self.apodization == 'Blackman-Harris':
            img_in_fft = np.fft.fft2(img_in*self.apodwindow)
        else :
            img_in_fft = np.fft.fft2(img_in)

        # only when all (circular) buffer positions have been filled
        if (self.totalframes >= self.Nbuf):
            # img_in is frame "t + dt"
            # frame in buffer is frame "t"

            #i_acc = self.totalframes // Nacc
            #_ISFaccum_list[i_acc] = _update_ISF_multiACCU(self, img_in_fft_real, img_in_fft_imag,_ISFaccum_list[i_acc])

            self._update_ISF(img_in_fft.real, img_in_fft.imag)
            self.ISFcount += 1

        # update index for writing incoming frame (FFT2) to circular buffer
        if (self.ix_in <= 0):
            self.ix_in = self.Nbuf
        self.ix_in -= 1

        # write fft  of incoming image to buffer slot ix_in
        # GPU: this should be done using buffers in GPU memory
        self.BUFreal[self.ix_in, :, :] = img_in_fft.real
        self.BUFimag[self.ix_in, :, :] = img_in_fft.imag

        # frame counter
        self.totalframes += 1

    def _update_ISF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """
        for n in range(self.Nbuf):
            ixc = (n + self.ix_in) % self.Nbuf
            dISF = (img_in_fft_real - self.BUFreal[ixc,:,:])**2 \
                  + (img_in_fft_imag - self.BUFimag[ixc,:,:])**2
            # add to total isf
            self._ISFaccum[n,:,:] += dISF

    def _update_ISF_multiACCU(self, img_in_fft_real, img_in_fft_imag,_ISFaccum):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """

        for n in range(self.Nbuf):
            ixc = (n + self.ix_in) % self.Nbuf
            dISF = (img_in_fft_real - self.BUFreal[ixc,:,:])**2 \
                  + (img_in_fft_imag - self.BUFimag[ixc,:,:])**2
            # add to total isf
            _ISFaccum[n,:,:] += dISF

        return _ISFaccum


    def ISF(self):
        """return a new matrix with the full, correctly oriented
        ('fftshift'-ed) ISF[t,y,x]"""
        assert self.ISFcount > 0, 'No ISF is available (not enough '\
                                  'image frames to fill buffer?)'

        ISFarray = np.zeros((self.Nbuf+1,self.Npx,self.Npx))
        # in ImageStructureFunction ISFarray, the array is offset by 1
        # this is because ISF[0] contains delta t = 0
        for ix, dISF in enumerate(self._ISFaccum):
            #ISFarray[ix+1,:,:] = np.fft.fftshift(dISF)/self.ISFcount
            ISFarray[ix+1,:,:] = dISF/self.ISFcount

        return ISFarray



class ImageStructureEngineLB5(ImageStructureEngineLB4):
    """Single-threaded numba-version of ImageStructureEngineLB4
    
    Tuned by Lancelot Barthe 
    """
    def _update_ISF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """
        # call numba
        numba_ISF_push_frame_fft_v1(img_in_fft_real, img_in_fft_imag,
                                    self._ISFaccum, self.ix_in,
                                    self.BUFreal, self.BUFimag,
                                    self.Nbuf)



class ImageStructureEngineLB6(ImageStructureEngineLB4):
    """Multi-threaded numba-version of ImageStructureEngineLB4
    
    Tuned by Lancelot Barthe 
    """
    def _update_ISF(self, img_in_fft_real, img_in_fft_imag):
        """process an incoming frame for the image structure function

        no checks are done on img_in, we trust that it is np.array of shape (Npx,Npx)
        of type np.float

        """
        # call numba
        numba_ISF_push_frame_fft_v2(img_in_fft_real, img_in_fft_imag,
                                   self._ISFaccum, self.ix_in,
                                   self.BUFreal, self.BUFimag,
                                   self.Nbuf)

        # !!! WATCH OUT !!! #
        # PASS PARAMETER V AND PHI

        #numba_ISF_push_frame_fft_v2_corr(img_in_fft_real, img_in_fft_imag, self._ISFaccum, self.ix_in,
        #self.BUFreal, self.BUFimag,self.Nbuf, 2*12.3, 0, self.ux, self.uy, self.tauf)



#%%


def _ImageStructureEngineSelector(n: int):
    """
    Return a ImageStructureEngine class, corresponding to a specific 'model' of
    ImageStructureEngine.



    Parameters
    ----------
    n : int
        The number of the model (or 'type') of ImageStructureEngine. Model 0
        and model 1 refer to the same Engine, the default, reference engine.

    Raises
    ------
    Exception
        In case the number of the model is not in the catalog, raise an
        exception.

    Returns
    -------
    ImageStructureEngine class
        A class of ImageStructureEngine that can be used to instantiate
        ImageStructureEngines.

    """
    if n==2:
        return ImageStructureEngine2
    elif n==3:
        return ImageStructureEngine3
    elif n==4:
        return ImageStructureEngine4
    elif n==5:
        assert not (numba is None), "This requires Numba to be installed."
        return ImageStructureEngine5
    elif n==6:
        assert not (numba is None), "This requires Numba to be installed."
        return ImageStructureEngine6
    elif n==14:
        return ImageStructureEngineLB4
    elif n==15:
        assert not (numba is None), "This requires Numba to be installed."
        return ImageStructureEngineLB5
    elif n==16:
        assert not (numba is None), "This requires Numba to be installed."
        return ImageStructureEngineLB6
    elif n==7:
        assert not (cuda is None), "This requires Numba and CUDA to be active."
        return ImageStructureEngine7
    elif n==0 or n==1:
        return ImageStructureEngine0
    else:
        raise Exception('This model of ImageStructureEngine does not exist')




def ImageStructureEngine(Npx, Nbuf, engine_model = 0, **kwargs):
    """
    Choose and initialize (instantiate) an ImageStructureEngine

    The current DDM Toolkit proposes several different algorithms and
    implementations for accumulating a DDM Image Structure Function. The
    object classes that contain these implementations are called
    `ImageStructureEngine`. The different implementations are referred to as
    different 'models' or 'types' of ImageStructureEngine. Each 'model'
    calculates the same image structure function, but using a different
    algorithm.

    Currently available ImageStructureEngine models:

    0, 1 : The default, standard, reference engine (pure numpy)
    2    : "All-FFTs-in-one" (pure numpy), not very useful, obsolete
    3    : "Buffered FFT" (pure numpy), fastest 'pure numpy'
    4    : "Basic 2nd gen" (pure numpy), 2nd-generation engine
    5    : "Basic 2nd gen" (numba CPU), single-threaded
    6    : "Multithread" (numba CPU)
    7    : "CUDA" (numba CUDA GPU)

    If `engine_model == -1`, then the 'best' available engine will be selected
    automaticall (fastest engine)

    In all cases, the FFTs are still done by numpy's standard (single thread)
    FFT2 code. More room for optimization here.


    Parameters
    ----------
    Npx : int
        Height and width (in pixels) of the (square) input image and
        corresponding image structure function.
    Nbuf : int
        Depth (in frames) of the image structure function (and the
        corresponding buffer)
    engine_model : int, optional
        Index of the particular model of ImageStructureEngine.
        The default is 0.
    **kwargs : keyword arguments
        Additional keyword parameters to be passed to the particular
        ImageStructureEngine class.

    Returns
    -------
    Instance of one of the ImageStructureEngine classes
        DESCRIPTION.

    """
    if engine_model == -1:
        engine_model = best_available_engine_model
    ISEclass = _ImageStructureEngineSelector(engine_model)
    return ISEclass(Npx, Nbuf, **kwargs)


#%%

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

    saveRadAvg => save the complete radially averaged ISF for later
    use via ImageStructureFunctionRadAvg

    TODO refactor ImageStructureFunction/ImageStructureFunctionRadAvg should
    derive from one and the same base class

    e.g.

    class ImageStructureFunctionBase:
        def __init__(self, hdic=None):
            etc. etc.
            (mainly handling hdic)

    class ImageStructureFunction(ImageStructureBase):
        etc. etc.
    """
    def __init__(self, ISFarray, tauf, ux, uy, hdic=None):
        self.isRadialAverage = False
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
            self.hdic = hdic

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
    def fromFile(cls, fpn):
        with np.load(fpn, allow_pickle=True) as npz:
            hdic = npz['header'].item()

            tauf = npz['tauf']
            uy = npz['uy']
            ux = npz['ux']
            ISF = npz['ISF']

            ISF1 = cls(ISF, tauf, ux, uy, hdic)

            ISF1.q = npz['q']
            ISF1.qx = npz['qx']
            ISF1.qy = npz['qy']
            ISF1.tau = npz['tau']

        ISF1.real_world_units = hdic['real_world_units']
        return ISF1

    @classmethod
    def fromImageStructureEngine(cls, ISE_instance):
        hdic ={}
        hdic['Npx'] = ISE_instance.Npx
        hdic['Nbuf'] = ISE_instance.Nbuf
        hdic['ISFcount'] = ISE_instance.ISFcount
        hdic['totalframes'] = ISE_instance.totalframes

        hdic['Npick'] = ISE_instance.Npick
        hdic['Navg'] = ISE_instance.Navg
        hdic['Ndrop'] = ISE_instance.Ndrop
        hdic['apodwindow'] = ISE_instance.apodwindow
        return cls(ISE_instance.ISF(), ISE_instance.tauf,
                   ISE_instance.ux, ISE_instance.uy,
                   hdic = hdic)


    def save(self, fpn):
        """
        Save the ImageStructureFunction to a file.

        This generates a file that is identical to (or at least compatible
        with) a file generated using ImageStructureEngine.saveISF method.


        Parameters
        ----------
        fpn : str
            File pathname.

        Returns
        -------
        None.

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
        hdic['real_world_units'] = self.real_world_units
        np.savez(fpn,
                 header = hdic,
                 ISF = self.ISF,
                 tauf = self.tauf, uy = self.uy, ux = self.ux,
                 q = self.q, qx = self.qx, qy = self.qy,
                 tau = self.tau
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


    def saveRadAvg(self, fpn):
        """Radially average the complete ISF and save it to an NPZ file,
        including the ISFengine settings.

        This 'save'
        method is intended to save the result of ISF calculation in a compact
        form in the case that we are dealing with isotropic particle motion.

        It can be opened using the ImageStructureFunctionRadAvg class

        .
        """
        # radial averager
        ISFqtau = np.zeros((len(self.tauf),len(self.u)))
        for i in range(len(self.tauf)):
            ISFqtau[i,:] = self.radavg(i)
        # remove "zero-frequency garbage" at center
        ISFqtau[:,0] = 0.0

        self.hdic['real_world_units'] = self.real_world_units
        np.savez(fpn,
                 header = self.hdic,
                 ISFRadAvg = ISFqtau,
                 tauf = self.tauf, u = self.u,
                 q = self.q, tau = self.tau
                 )



class ImageStructureFunctionRadAvg:
    """Class containing a radially averaged ISF in the form of an array
    together with scaling information

    In its radially averaged form, the ISF takes much less space, and is more
    easily transferred for analysis.

    TODO: This class and the basic ImageStructureFunction class should be
    refactored, so that they both derive from the same base class. At present
    (for ease of programming) they are two distinct classes. Actually, the
    ImageStructureFunctionRadAvg was obtained by copying the code of
    ImageStructureFunction and then modifying it

    Can be constructed by loading a suitable NPZ file (output of
    ImageStructureEngine.saveRadAvg)

        isf = ImageStructureFunctionRadAvg.fromFile(NPZfilename)

    The class can NOT be instantiated directly from an
    ImageStructureEngine
    """
    def __init__(self, ISFRadAvg_array, tauf, u, hdic=None):
        self.isRadialAverage = True
        self.ISFqtau = ISFRadAvg_array
        self.tauf = tauf
        self.u = u

        self.Npx = len(self.u)

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

        #initialize this to pixel, frame units
        # replace with 'real world' units by calling
        #   real_world(um_p_px, s_p_frm)
        self.real_world_units = False
        self.q = 2*np.pi*self.u
        self.tau = self.tauf * 1.0


    @classmethod
    def fromFile(cls, fpn):
        with np.load(fpn, allow_pickle=True) as npz:
            hdic = npz['header'].item()
            tauf = npz['tauf']
            u = npz['u']
            ISFRadAvg = npz['ISFRadAvg']

            RAISF = cls(ISFRadAvg, tauf, u, hdic)

            RAISF.q = npz['q']
            RAISF.tau = npz['tau']
        RAISF.real_world_units = hdic['real_world_units']
        return RAISF


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
        self.tau = self.tauf * s_p_frm
        self.real_world_units = True
