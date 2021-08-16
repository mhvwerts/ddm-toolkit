============
Some details
============



ImageStructureEngine
====================

Pre-processing: Picking, averaging, dropping
--------------------------------------------
Using the 'pick', 'avg', 'drop' keyword parameters, versatility is introduced for processing video sequences more efficiently, or reducing noise. This may likely also find use for implementing parallel processing schemes (several ImageStructureEngine instances, each on its own thread). These parameters change the behaviour of the frame pre-processing, which is done before calling the actual ISF calculation.

We only propose frame averaging, since simple frame summing was not found to be useful.


Pre-processing: apodization (windowing)
---------------------------------------
Another option offered by the pre-processor is apodization (also known as 'windowing'), using the Blackman-Harris windowing function. This windowing was suggested by Giavazzi et al. (*Eur. Phys. J. E* **2017**, *40*, 97. `DOI link 1`_ ).

.. _DOI link 1: https://dx.doi.org/10.1140/epje/i2017-11587-3


Simulation parameters
=====================

The system
----------
::

    bl_x    box width (x length)               [world units; µm]	
    bl_y    box height (y length)              [world units; µm]
    Np      number of particles                [particles]
    D       diffusion coefficient              [world units; µm2 s-1]
    

Dynamics simulation
-------------------
::
    
    T       total time                         [world units; s]
    Nt      number of time steps               [frames]
    (later: sampling settings?)
    

Image synthesis
---------------
::

    w       Gaussian spot radius               [world units; µm]
            (microscope resolution)
    im_Nx   image width in number of pixels    [pixels]
    im_Ny   image height in number of pixels   [pixels]
    im_x0   viewport, left x coord             [world units; µm]   
    im_y0   viewport, bottom y coord           [world units; µm]
    im_x1   viewport, right x coord            [world units; µm]   
    im_y1   viewport, top y coord              [world units; µm]


Derived quantities
------------------
These will be needed to convert the results of the DDM analysis of the
synthetic image sequences to real world units. They can be calculated
from the simulation input parameters. These quantities are (probably?)
the only ones that need to be transferred to the DDM analysis, together
with generated synthetic image sequence. This makes sense, since these
are the only experimental parameters that we have at our disposition
in a real-world experiment.


time resolution
...............
::

    dt=(T/Nt)               frame period [seconds per frame]

spatial resolution
..................
::

    dx=(im_x1-im_x0)/im_Nx  x image resolution [µm per pixel]
    dy=(im_y1-im_y0)/im_Ny  y image resolution [µm per pixel]
    
Typically, dx=dy



tifffile
========

If `Christoph Gohlke's 'tifffile'`_ Python package has been installed, ``ddm-toolkit`` will use that version, because it is likely the more recent version. (Our tip: use conda + Conda-forge for installing packages).

.. _Christoph Gohlke's 'tifffile': https://github.com/cgohlke/tifffile

``ddm-toolkit`` includes a copy of a legacy version of ``tifffile``, that will be used if a system ``tifffile`` is not available. See: `https://github.com/mhvwerts/tifffile`_

.. _https://github.com/mhvwerts/tifffile: https://github.com/mhvwerts/tifffile

In certain cases, a huge speed-up for decoding TIFF using the legacy ``tifffile`` is obtained by including a compiled C function. In order to compile it in your favorite environment, go to ``./ddm_toolkit/misc/tifffile_fork/`` and run ``python build_c.py build_ext --inplace``. This will generate a compiled binary module that is used by ``tifffile`` to speed up TIFF decoding.


