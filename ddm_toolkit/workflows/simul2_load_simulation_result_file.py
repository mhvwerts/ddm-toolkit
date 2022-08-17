#
import numpy as np
#
def simul2_load_simulation_result_file(fpn):
    """
    Load simulation file and parameters

    Parameters
    ----------
    fpn : str
        file pathname of the file to be written.

    Returns
    -------
    vid : list of 2D np.array (or 3D np.array)
        stack of synthetic video frames.
    params : DDMParams
        Object containing DDM Toolkit parameters

    """

    print('Loading file:', fpn)

    simulfile = np.load(fpn,
                        allow_pickle = True)
    vid = simulfile['videostack']

    params = simulfile['ddmparams'][()] # Ninja code to get object back.
                            # see: https://stackoverflow.com/a/8362451
                            #      https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
    simulfile.close()

    # set overdrive parameter (boost brightness)
    #TODO in parameter file?
    params.img_overdrive = 1.7

    return vid, params
