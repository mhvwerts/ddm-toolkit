#
import numpy as np
#
from ddm_toolkit.tifftools import TiffFile, tifffile_version
#
from ddm_toolkit import tqdm
#
from ddm_toolkit.utils import ROI_mask
#
def video1_load_preview(pars):
    '''

    Load a couple of frames of the file infp to realise a preview later on

    Parameters
    ----------
    pars : DDMParams
        Object holding all processing parameters.

    Returns
    -------
    vid : Array numpy
        Video to be dislayed into preview video player with darkening of RONI.

    '''

    # get video info
    print('file being processed: ', pars.infp)
    with TiffFile(pars.infp) as tif:
        tiflen = len(tif.pages) - 1 # remove last frame (sometime causes problems)
        print('TIFF length: {0:d} frames'.format(tiflen))
        # get first image in file in order to determine dimension
        img = tif.pages[0].asarray()
        print('frame shape: {0:d} x {1:d}'.format(*img.shape))

    # Extract ROI parameter from pars struct
    ROI_x1 = pars.ROI_x1
    ROI_y1 = pars.ROI_y1
    ROI_size = pars.ROI_size

    # enable full image processing
    if ROI_size < 0:
        ROI_x2 = img.shape[1]
        ROI_y2 = img.shape[0]
    else :
        ROI_x2 = ROI_x1+ROI_size
        ROI_y2 = ROI_y1+ROI_size

    # Extract frame parameters from paars file
    frm_start = pars.frm_start
    frm_Npreview = pars.frm_Npreview
    frm_prevend = frm_start + frm_Npreview
    if frm_prevend > tiflen:
        frm_prevend = tiflen
    frm_Npreview = frm_prevend - frm_start

    vidshape = (frm_Npreview,img.shape[0],img.shape[1])
    vid = np.zeros(vidshape)

    print('loading preview frames into memory')
    with TiffFile(pars.infp) as tif:
        for i in tqdm(range(frm_Npreview)):
            img = tif.pages[frm_start + i].asarray()

            # Realize a darkening of RONI
            mask = ROI_mask(ROI_x1, ROI_x2, ROI_y1, ROI_y2, img.shape)

            # copy all pixels divided by ROIcontrast (low intensity)
            vid[i,:,:] = img[:,:]*mask # / ROIcontrast

            ## only copy ROI zone at full intensity
            ## vid[i,ROI_y1:ROI_y2,ROI_x1:ROI_x2] = img[ROI_y1:ROI_y2,ROI_x1:ROI_x2]

    return vid
