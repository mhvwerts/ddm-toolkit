===============================================================
ddm_toolkit: Python Toolkit for Differential Dynamic Microscopy
===============================================================

------------
Introduction
------------
Dynamic differential microscopy (DDM) is a digital video-microscopic technique for studying the dynamics of microscopic systems, such as small particles in a liquid. See, *e.g.*,: [1] Cerbino and Trappe, *Phys. Rev. Lett.* **2008**, *100*, 188102; [2] Giavazzi et al. *Phys. Rev. E* **2009**, *80*, 031403; [3] Germain et al. *Am. J. Phys.* **2016**, *84*, 202.

DDM relies on numerical processing of videomicroscopy image sequences using specific computer code.In our lab, we are currently exploring DDM for nanoparticle characterization in microfluidics. This Github repository centralizes the contributions made to our lab's DDM Python code by the students and investigators who participate in this research. The aim is to constitute a standardized toolkit that students and colleagues can download and use for DDM. At present, the toolkit is in a preliminary and sparsely documented state. Do not hesitate to contact us if you are interested in using it and helping develop it further.

Our DDM toolkit has several specific original features:

- The DDM processing algorithm can process arbitrarily long input video streams, with memory requirements only determined by the duration (number of time lags) of the ISF. The individual video frames are read one-by-one from the video file on disk and then processed. The processing 'engine' accumulates an ISF, with each incoming frame adding to the overall ISF.
- The toolkit works with standard (OME) TIFF image-stacks via Gohlke's ``tifffile.py``.
- It includes basic Brownian simulation and video synthesis routines for generating idealized, artificial video-streams that can be used to test any specific DDM processing.
- The combination of simulation and processing enables us to cross-check the correct implementation of both the simulation and the DDM analysis.


------------
Installation
------------

There is no specific installable package yet for this toolkit, and it does not need separate installation. In order to use the toolkit, download this Github repository as a ZIP file (or 'clone' the repository) and unpack it in a separate folder on your computer. The scripts can be run from the command line using a suitable Python environment. We recommend Anaconda Miniconda3 with the `conda-forge`_ channel. The toolkit requires Python 3.6 or higher, and only needs ``numpy``, ``scipy``, ``matplotlib``. We also recommend that you install `C. Gohlke's tifffile`_ (``conda install tifffile``) and the latest version of `tqdm`_. (``conda install tqdm``)

.. _C. Gohlke's tifffile: https://github.com/cgohlke/tifffile
.. _tqdm: https://tqdm.github.io/

-----------
Basic usage
-----------
We present here a step-by-step approach to the generation and processing of DDM data: each step is carried out by a short custom-written Python script that uses the functionality from the ``ddm_toolkit`` module, and is called from the command-line interface (CLI). The result of each step is written to a data file, which is then passed to the script that carries out the next step. Since some of the steps take quite some processing time, the step-by-step approach avoids that you need to wait long minutes each time you run your script after having made only minor changes.

In addition to the CLI workflow, we are now actively developing Jupyter Notebooks that directly use the functions from the ``ddm_toolkit`` module (*i.e.* the ``ddm_toolkit`` does the heavy lifting, the Notebooks controls the analysis and display the results). Jupyter Notebooks allow for more interactivity, and enable direct documentation of the analyses carried out. Here, in this ``README`` we focus on the basic CLI workflow.

In order to get an idea of this CLI workflow, you may run a sequence which first generates a simulated video stack of 2D Brownian motion. This synthetic video is subsequently processed by the DDM algorithm, and analyzed in terms of the standard Brownian model.

You will need to use a command prompt in your favorite Python 3 environment (we recommend Anaconda Python `distribution`_ , and then tuning Conda to the `conda-forge`_ channel).

.. _distribution: https://www.anaconda.com/products/individual
.. _Conda-forge: https://conda-forge.org/



Two-dimensional Brownian motion of a set of particles is generated and converted to synthetic videomicroscopy frames by running the first script. The image stack is stored in ``datafiles``.

.. code-block::

   python simul1_simulate_synthesize.py

You can then visualize the simulated video using:

.. code-block::

   python simul2_inspect_video.py


Then, calculation of the ISF from the simulated video:

.. code-block::

    python simul3_ISF_from_simulation.py


This ISF can be displayed as a video:

.. code-block::

    python xDDM1_inspect_ISF.py


And finally, the ISF is analyzed in terms of the simple Brownian model:

.. code-block::

    python xDDM2_analyze_brownian_model.py


The simulation and analysis parameters required by the scripts are set using using (default) parameter settings from the code (these can be found in ``ddm_toolkit.parameters``). Alternative parameter sets can be supplied as text files by using their filename as an argument to the Python script. For example:

.. code-block::

    python simul1_simulate_synthesize.py simul0_params_example.txt
    # etc.


Execution of this sequence of CLI scripts is a basic way of testing the code base. 


-----
TO DO
-----

- Document processing of real video files (scripts, Notebooks)
- Organize documentation (we have set up `Sphinx`_ in the ``.\doc`` directory)

.. _Sphinx: https://www.sphinx-doc.org


---------------
Other DDM codes
---------------

There are several other DDM codes available. Our DDM toolkit aims to be a standardized package for use internally in our lab. It aims also to provide a simple Python-based, generic, extensible toolkit that can be used for testing, benchmarking and comparing different approaches.

- `cddm (Python)`_ by Petelin and Arko (documented toolkit, also for cross-DDM)
- `diffmicro (C++/CUDA)`_ by Cerchiari et al. (fast DDM algorithms with/without GPU)
- `quickDDM (Python)`_ by Symes and Penington (GPU acceleration)
- `Differential-Dynamic-Microscopy---Python`_ by McGorty et al. (stack of Python notebooks)
- `DDM (Matlab; Python notebook)`_ by Germain, Leocmach, Gibaud
- `DDMcalc (Matlab)`_ by Helgeson et al.
- `ConDDM (C++ source)`_ by Lu et al. (for confocal DDM, CUDA, 2012)

.. _cddm (Python): https://github.com/IJSComplexMatter/cddm
.. _diffmicro (C++/CUDA): https://github.com/giovanni-cerchiari/diffmicro
.. _DDMcalc (Matlab): https://sites.engineering.ucsb.edu/~helgeson/ddm.html
.. _DDM (Matlab; Python notebook): https://github.com/MathieuLeocmach/DDM
.. _quickDDM (Python): https://github.com/CSymes/quickDDM
.. _Differential-Dynamic-Microscopy---Python: https://github.com/rmcgorty/Differential-Dynamic-Microscopy---Python
.. _ConDDM (C++ source): https://github.com/peterlu/ConDDM



-----------
Development
-----------

This toolkit is being maintained and developed by Martinus Werts (CNRS and ENS Rennes, France). It contains contributions from Lancelot Barthe (ENS Rennes), Nitin Burman (IISER Mohali, India), Jai Kumar (IISER Bhopal, India), Greshma Babu (IISER Bhopal) and Ankit Lade (IISER Bhopal). Suzon Pucheu, Elias Abboubi and Pierre Galloo-Beauvais (ENS Rennes) did further testing and application. The students from IISER worked on DDM during their research projects at ENS Rennes in the context of the IISER-ENS exchange program.


Python version requirement and dependencies
===========================================
Python 3.6 or newer is needed to run all of the code. The aim is to have a monolithic code-base that depends only on Python 3.x, its standard modules, and ``numpy``, ``scipy`` and ``matplotlib``. Any other external modules that we use (``tifffile``, ``tqdm`` and ``videofig``) have been directly incorporated ("assimilated") by copying their source code into the ``ddm_toolkit`` code tree. However, if ``tqdm`` and/or ``tifffile`` are available on the system, these (probably more recent) modules will be used.


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
A rudimentary code testing infrastructure is in place, using `pytest`_. See the `README file in the tests directory`_ for further information

.. _pytest: https://docs.pytest.org/en/stable/
.. _README file in the tests directory: ./tests/README.rst



