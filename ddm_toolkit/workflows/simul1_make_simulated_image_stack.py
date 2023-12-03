import numpy as np

from ddm_toolkit import tqdm
from ddm_toolkit.simulation import ParticleSim2D
from ddm_toolkit.simulation import ImageSynthesizer2D



def simul1_make_simulated_image_stack(dparams):
    """
    Do a basic Brownian simulation and generate corresponding synthetic video

    Parameters
    ----------
    dparams : DDMParams
        object holding all simulation parameters.

    Returns
    -------
    ims : list of 2D np.array
        stack of synthetic video frames.

    """
    psimul = ParticleSim2D(dparams)
    imgsynthez = ImageSynthesizer2D(psimul)
    Nframes = dparams.sim_Nt
    Npx = dparams.sim_img_Npx
    ims = np.zeros((Nframes,Npx,Npx),
                   dtype = np.float64)
    for ix in tqdm(range(Nframes)):
        img = imgsynthez.get_frame(ix)
        ims[ix,:,:] = img
    return ims
