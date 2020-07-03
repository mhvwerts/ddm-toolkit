===============================================================
ddm_toolkit: Python Toolkit for Differential Dynamic Microscopy
===============================================================


------------
Introduction
------------
In our lab, we are currently exploring differential dynamic microscopy (DDM) for nanoparticle characterization in microfluidics. DDM relies on numerical processing of videomicroscopy image sequences using specific computer code. This Github repository centralizes the contributions made to our lab's DDM Python code by the students and investigators who participate in this research. The aim is to constitute a toolkit that students and colleagues can download and use whenever they feel like doing DDM. At present, the toolkit is in a preliminary and sparsely documented state, "mais a le mérite d'exister" (as we say in France). Do not hesitate to contact us if you are interested in using it and helping develop it further.

Our DDM toolkit has several specific original features:

- The DDM processing algorithm can process arbitrarily long input video streams, with memory requirements only determined by the duration (number of time lags) of the ISF. The individual video frames are read one-by-one from the video file on disk and then processed. The processing 'engine' accumulates an ISF, with each incoming frame adding to the overall ISF.
- The algorithm provides several pre-processing options (averaging of frames, picking one frame for every N incoming frames, skipping frames at the start, apodization of incoming images).
- The toolkit works with standard (OME) TIFF image-stacks thanks to inclusion of Gohlke's ``tifffile.py``.
- It includes basic Brownian simulation and video synthesis routines for generating idealized, artificial video-streams that can be used to test any specific DDM processing.
- The combination of simulation and processing enables us to cross-check the correct implementation of both the simulation and the DDM analysis.


------------
Installation
------------

There is no specific installable package yet for this toolkit, and it does not need separate installation. In order to use the toolkit, download this Github repository as a ZIP file (or 'clone' the repository) and unpack it in a separate folder on your computer. The scripts can be run from the command line using a suitable Python environment. We recommend Anaconda Miniconda3 with the `conda-forge`_ channel. The toolkit requires Python 3.6 or higher, and needs ``numpy``, ``scipy``, ``matplotlib`` and ``lmfit``.


-----------
Basic usage
-----------
We encourage following a step-by-step approach to the generation and processing of DDM data: each step is carried out by a short custom-written Python script that uses the functionality from the ``ddm_toolkit`` module. The result of each step is written to a data file, which is then passed to the script that carries out the next step. Since some of the steps take quite some processing times, the step-by-step approach avoids that you need to wait long minutes each time you run your script after having made only minor changes.

In order to get an idea of the work flow, you may run a sequence which first generates a simulated video stack of 2D Brownian motion. This synthetic video is subsequently processed by the DDM algorithm, and analyzed in terms of the standard Brownian model.

You will need to use a command prompt in your favorite Python 3 environment (we recommend Anaconda Python `distribution`_ , and then tuning Conda to the `conda-forge`_ channel).

.. _distribution: https://www.anaconda.com/products/individual
.. _Conda-forge: https://conda-forge.org/



Two-dimensional Brownian motion of a set of particles is generated and converted to synthetic videomicroscopy frames by running the first script. The image stack is stored in ``datafiles``.

.. code-block::

   python simul1_simulate_synthesize.py

You can then visualize the generated video using:

.. code-block::

   python simul2_inspect_video.py


Then, calculation of the ISF and visual inspection:

.. code-block::

    python simul3_calculate_ISF
    python simul4_inspect_ISF.py


And finally, analysis of the ISF in terms of the simple Brownian model:

.. code-block::

    python simul5_analyze.py


The simulation and analysis parameters required by the scripts are set using a (default) parameter file ``simul0_default_params.txt``. Alternative parameter files may be used by supplying their filename as an argument to the Python script.

Execution of this sequence of scripts is a way of testing the code base, in a basic and somewhat cumbersome fashion. A nice aspect is that intermediate results (synthetic video, ISF) are stored in the ``datafiles`` directory, so that it is not necessary to re-calculate them over and over when experimenting with a script or an analysis.


-----
TO DO
-----

- Add an example of processing a real video file
- (Re)organize documentation: put certain sections in separate files


---------------
Other DDM codes
---------------

There are several other DDM codes available. Our DDM toolkit aims to be a standardized package for use internally in our lab. It aims also to provide a simple Python-based, generic, extensible toolkit that can be used for testing, benchmarking and comparing different approaches.

- `quickDDM (Python)`_ by Symes and Penington (GPU acceleration)
- `Differential-Dynamic-Microscopy---Python`_ by McGorty et al. (stack of Python notebooks)
- `DDM (Matlab; Python notebook)`_ by Germain, Leocmach, Gibaud
- `DDMcalc (Matlab)`_ by Helgeson et al.
- `ConDDM (C++ source)`_ by Lu et al. (for confocal DDM, CUDA, 2012)

.. _DDMcalc (Matlab): https://sites.engineering.ucsb.edu/~helgeson/ddm.html
.. _DDM (Matlab; Python notebook): https://github.com/MathieuLeocmach/DDM
.. _quickDDM (Python): https://github.com/CSymes/quickDDM
.. _Differential-Dynamic-Microscopy---Python: https://github.com/rmcgorty/Differential-Dynamic-Microscopy---Python
.. _ConDDM (C++ source): https://github.com/peterlu/ConDDM



-----------
Development
-----------

This toolkit is being maintained and developed by Martinus Werts (CNRS and ENS Rennes, France). It contains further contributions from Lancelot Barthe (ENS Rennes), Nitin Burman (IISER Mohali, India), Jai Kumar (IISER Bhopal, India) and Greshma Babu (IISER Bhopal, India). The students from IISER worked on DDM during their research projects at ENS Rennes in the context of the IISER-ENS exchange program.


Python version requirement and dependencies
===========================================
Python 3.6 or newer is needed to run all of the code: we introduced some static type checking, here and there. We did not test with older versions of Python.

The aim is to have a monolithic code-base that only depends on Python 3.x, its standard modules, and ``numpy``, ``scipy`` and ``matplotlib``. Any other external modules that we use (currently: the brilliant ``tifffile``, the lovely ``tdqm`` and the nice ``python-tabular``) have been directly incorporated ("assimilated") by copying their source code into the ``ddm_toolkit`` code tree.

There is one extra external dependency at the moment, `LMFit`_ (which depends on yet other external packages...). Since our fitting needs are simple, we may consider simply using SciPy's `curve_fit`, in order to minimize dependence on external modules.

.. _LMFit: https://lmfit.github.io/lmfit-py/



Vocabulary
==========
In our choice of terms, we aim to be consistent with common usage in the existing DDM literature. In our text, we use the term "image structure function" (ISF) both for the (differential) image structure function at a certain time lag AND for the complete sequence of (differential) image structure functions over a series of time lags. We would have preferred to call the latter "video structure function" (which would be 2D spatial + time)


Programming style
=================
We are scientists, not programmers. However, we intend to adopt good programming habits, that will enable our programs to be used with confidence by other scientists. Good habits include documenting our code, coding cleanly and understandably, close to the mathematical formulation of the science. They also include providing tests for our code. 

The adoption of good programming habits should be considered work-in-progress!

We use numpy-style docstrings, even though we are not yet 100% compliant.

An important way of testing scientific software is to use it on well-defined test cases whose results are known ("benchmarks").


Code testing
============
A very rudimentary code testing infrastructure is in place, using `pytest`_. See the `README file in the tests directory`_ for further information

.. _pytest: https://docs.pytest.org/en/stable/
.. _README file in the tests directory: ./tests/README.rst





------------------------------
Documentation: further details
------------------------------

ImageStructureEngine
====================
Pre-processing: Picking, averaging, dropping
--------------------------------------------
Using the 'pick', 'avg', 'drop' keyword parameters, versatility is introduced for processing video sequences more efficiently, or reducing noise. This may likely also find use for implementing parallel processing schemes (several ImageStructureEngine instances, each on its own thread). These parameters change the behaviour of the frame pre-processing, which is done before calling the actual ISF calculation.

We only propose frame averaging, since simple frame summing was not found to be useful. Thus: picking and/or averaging


Pre-processing: apodization (windowing)
---------------------------------------
Another option offered by the pre-processor is apodization (also known as 'windowing'), using the Blackman-Harris windowing function. This windowing was suggested by Giavazzi et al. (*Eur. Phys. J. E* **2017**, *40*, 97. `DOI link 1`_ ).

.. _DOI link 1: https://dx.doi.org/10.1140/epje/i2017-11587-3


Simulation input parameters
===========================

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

--------
tifffile
--------


This toolkit contains a 'hard' copy of a fork of Christoph Gohlke's 'tifffile', for reading TIFF image sequences. See: `https://github.com/mhvwerts/tifffile`_

.. _https://github.com/mhvwerts/tifffile: https://github.com/mhvwerts/tifffile

In certain cases, a huge speed-up for decoding TIFF is obtained by including a compiled C function. In order to compile it in your favorite environment, go to ``./tifffile/`` and run ``python build_c.py build_ext --inplace``. This will generate an importable library file.


