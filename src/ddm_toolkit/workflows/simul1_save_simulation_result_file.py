import numpy as np


def simul1_save_simulation_result_file(fpn, videostack, ddmparams):
    """
    Write the result of a simulation to a file

    Parameters
    ----------
    fpn : str
        file pathname of the file to be written.
    videostack : list of 2D np.array (or 3D np.array)
        stack of synthetic video frames.
    ddmparams : DDMParams
        object holding all simulation parameters.

    Returns
    -------
    None.

    """
    print("Writing NPZ file with video and simulation parameters...")
    np.savez_compressed(fpn,
                        videostack = videostack,
                        ddmparams = ddmparams)
