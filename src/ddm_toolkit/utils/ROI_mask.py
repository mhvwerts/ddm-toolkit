#
import numpy as np
#
def ROI_mask(ROI_x1, ROI_x2, ROI_y1, ROI_y2, img_shape,RONI_val = 0.75):
    '''


    Parameters
    ----------
    ROI_x1 : INT
        Lowest x-coordinates of ROI.
    ROI_x2 : INT
        Highest x-coordinates of ROI.
    ROI_y1 : INT
        Lowest y -coordinates of ROI.
    ROI_y2 : INT
        Highest y-coordinates of ROI.
    img_shape : TUPLE
        Image shape (x,y).
    RONI_val : FLOAT
        Mask value to be applied in RONI (in 0,1)

    Returns
    -------
    mask : ARRAY (Numpy) Optionnal
        Mask value used to darken RONI.

    '''
    # Compute RONI mask
    mask = np.ones((img_shape[1],img_shape[0]))

    for k in range(0,img_shape[1]):
        for l in range(0,img_shape[0]):

            if (k < ROI_x1 or k > ROI_x2) or (l < ROI_y1 or l > ROI_y2) :
                mask[k,l] = RONI_val

    return mask
