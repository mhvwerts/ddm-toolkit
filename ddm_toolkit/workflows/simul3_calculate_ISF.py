#
from ddm_toolkit import tqdm
#
from ddm_toolkit.ddm import ImageStructureEngine
from ddm_toolkit.ddm import ImageStructureFunction
#
def simul3_calculate_ISF(ims, params):
    """
    Calculate Image Structure Function from a simulated video

    Parameters
    ----------
    ims : list of 2D np.arrays (or a 3D np.array)
        Stack of video frames.
    params : DDMParams
        Object containing DDM Toolkit parameters.

    Returns
    -------
    ISF1 : ImageStructureFunction
        Data structure containing the Image Structure Function and relevant
        parameters.

    """


    ISE1 = ImageStructureEngine(params.ISE_Npx, params.ISE_Nbuf,
                                engine_model = params.ISE_type)

    for it in tqdm(range(params.Nframes)):
        ISE1.push(ims[it])

    ISF1 = ImageStructureFunction.fromImageStructureEngine(ISE1)

    # set real world units
    ISF1.real_world(params.um_p_pix, params.s_p_frame)

    return ISF1
