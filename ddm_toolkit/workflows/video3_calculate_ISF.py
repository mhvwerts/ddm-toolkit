#
from ddm_toolkit.tifftools import TiffFile, tifffile_version
#
from ddm_toolkit import tqdm
#
import numpy as np
#
from ddm_toolkit.ddm import ImageStructureEngine
from ddm_toolkit.ddm import ImageStructureFunction
#
from ddm_toolkit.ddm import available_engine_models
#
def video3_calculate_ISF(params):
    """
    Calculate Image Structure Function from a real TIFF video stack

    Parameters
    ----------
    params : DDMParams
        Object containing DDM Toolkit parameters.

    Returns
    -------
    ISF1 : ImageStructureFunction
        Data structure containing the Image Structure Function and relevant
        parameters.

    """

    # TO DO :
        # It would be nicer to replace the TIFF file name with a generalized
        # video object that stream frames one by one.


    print('Video-file being processed: ', params.videofpn)
    print('tifffile version: ', tifffile_version)
    tif = TiffFile(params.videofpn)

    tiflen = len(tif.pages) - 1 # remove last frame (sometime causes problems)
    print('TIFF length: {0:d} frames'.format(tiflen))

    # first image to get dimensions
    imgfull = tif.pages[0].asarray()
    # enable full image processing
    if params.ROI_size < 0:
        params.ROI_x1 = 0
        params.ROI_y1 = 0
        params.ROI_x2 = imgfull.shape[1]
        params.ROI_y2 = imgfull.shape[0]
        params.ROI_size = imgfull.shape[1]
    else :
        params.ROI_x2 = params.ROI_x1 + params.ROI_size
        params.ROI_y2 = params.ROI_y1 + params.ROI_size

    params.ISE_Npx = params.ROI_size

    #print('ROI: x={0:d}...{1:d}  y={2:d}...{3:d}'.format(params.ROI_x1,
    #                                                     params.ROI_x2,
    #                                                     params.ROI_y1,
    #                                                     params.ROI_y2))

    if (params.frm_end > tiflen) or (params.frm_end < 1):
        params.frm_end = tiflen

    print('selected frames: start={0:d}  end={1:d}'.format(params.frm_start,
                                                           params.frm_end))
    Nframes = params.frm_end - params.frm_start

    #TODO: further configure engine
    print('Engine used n° : ',params.ISE_type)
    ISE = ImageStructureEngine(params.ISE_Npx, params.ISE_Nbuf,
                                engine_model = params.ISE_type, apodization = params.apodization)

    # loop over selected images in file
    # tqdm just makes a progress bar (useful for large files!)
    for i in tqdm(range(Nframes)):
        imgfull = tif.pages[i + params.frm_start].asarray()
        # get ROI only
        img = imgfull[params.ROI_y1:params.ROI_y2,
                      params.ROI_x1:params.ROI_x2]
        ISE.push(img)

    tif.close()

    print()
    print('#frames contributing to averaged ISF (ISFcount): {0:d}'.format(ISE.ISFcount))

    ISF = ImageStructureFunction.fromImageStructureEngine(ISE)

    # set real world units
    ISF.real_world(params.um_p_pix, params.s_p_frame)

    return ISF
