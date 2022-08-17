#
from ddm_toolkit import tqdm
#
import numpy as np
#
from ddm_toolkit.tifftools import TiffFile, tifffile_version
#
from ddm_toolkit.utils import ROI_mask
#
def video2_check(pars):
    '''
    Check videos quality by computing Intensity and energy of frames

    Parameters
    ----------
    pars : DDMParams
        Object holding all processing parameters

    Returns
    -------
    imgItot : ARRAY (Numpy)
        Total intensity of every frame.
    E_dimg : ARRAY (Numpy)
        Difference in Energy between frames.
    tiflen : INT
        Length of the video (tif file).

    '''

    print('file being processed: ', pars.infp)

    with TiffFile(pars.infp) as tif:
        tiflen = len(tif.pages) - 1 # remove last frame (sometime causes problems)
        print('TIFF length: {0:d} frames'.format(tiflen))

        # first image to get dimensions
        imgfull = tif.pages[0].asarray()

    ## enable full image processing
    # Extract ROI parameter from pars struct
    ROI_x1 = pars.ROI_x1
    ROI_y1 = pars.ROI_y1
    ROI_size = pars.ROI_size

    # enable full image processing
    if ROI_size < 0:
        ROI_x2 = imgfull.shape[1]
        ROI_y2 = imgfull.shape[0]
    else :
        ROI_x2 = ROI_x1+ROI_size
        ROI_y2 = ROI_y1+ROI_size

    print('ROI: x={0:d}...{1:d}  y={2:d}...{3:d}'.format(ROI_x1,
                                                         ROI_x2,
                                                         ROI_y1,
                                                         ROI_y2))

    frm_start = pars.frm_start
    frm_end = pars.frm_end
    if (frm_end > tiflen) or (frm_end < 1):
        frm_end = tiflen

    print('selected frames: start={0:d}  end={1:d}'.format(frm_start,
                                                           frm_end))
    Nframes = frm_end - frm_start

    imgItot = np.zeros(Nframes)
    E_dimg = np.zeros(Nframes-1)
    previmg = 0.0 # just initialize this as a scalar value

    with TiffFile(pars.infp) as tif:
        # loop over selected images in file
        # tqdm just makes a progress bar (useful for large files!)
        for i in tqdm(range(Nframes)):
            imgfull = tif.pages[i + frm_start].asarray()
            # Realize a darkening of RONI
            mask = ROI_mask(ROI_x1, ROI_x2, ROI_y1, ROI_y2, imgfull.shape, RONI_val = 0)
            img = imgfull[:,:]*mask
            # The total intensity is simply the sum over all
            # pixels in an image frame
            imgItot[i] = np.sum(img)
            # Here we calculate the difference between two
            # subsequent frames (frame pair #0 = frame#0 and frame#1)
            # and then take the 'energy' (sum over squares)
            # This helps in detecting any glitches in the video (skipped
            # frames etc.)
            if i>0:
                dimg = (img - previmg)
                E_dimg[i-1] = np.sum(dimg**2)
                previmg = img
            else:
                previmg = img

    return imgItot, E_dimg, tiflen
