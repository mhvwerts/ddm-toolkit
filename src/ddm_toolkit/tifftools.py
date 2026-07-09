# -*- coding: utf-8 -*-

import numpy as np

# tifffile aliases (DDM toolkit depends on Gohlke's tifffile package)
from tifffile import TiffFile
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite
from tifffile import TiffWriter
from tifffile import __version__ as tifffile_version

# print('tifffile version:', tifffile_version)




# additional utility functions for Ximea OME-TIFF file

def ximea_ome_tiff_metadata(fp):
    """
    Get OME-TIFF XML metadata and frame time-stamps from Ximea file

    Parameters
    ----------
    fp : str or pathlib.Path
        path to file.

    Returns
    -------
    omexmlstr : string
        All XML data.
    timestamps : np.array
        Time-stamps for all frames (in µs).

    """
    video_OK = False
    
    with TiffFile(fp, is_ome=True) as tff:
        omexmlstr = tff.ome_metadata
        
        for k in tff.pages[0].tags.keys():
            print(k, tff.pages[0].tags[k])
            
        pnl = []
        for pg in tff.pages:
            pnl.append(int(pg.tags[285].value))
            
        timestamps = np.array(pnl, dtype=int)
    
    print()
    avg_fps = 1/((timestamps[-1]-timestamps[0])/(len(timestamps)-1)*1e-6)
    print(f'Average FPS: {avg_fps}')
    
    dt = (timestamps[1:]-timestamps[0:-1])*1e-3
    avg_spf = np.mean(dt)
    max_spf = dt.max()
    max_spf_pos = dt.argmax()
    min_spf = dt.min()
    min_spf_pos = dt.argmin()
    print(f'Average frame-period: {avg_spf:.3f} ms')
    print(f'Maximal frame-period: {max_spf:.3f} ms after frame #{max_spf_pos:d}')
    print(f'Minimal frame-period: {min_spf:.3f} ms after frame #{min_spf_pos:d}')
    
    fluct_pct = ((max_spf - min_spf)/avg_spf)*100.
    print(f'Period fluctuation {fluct_pct:.2f} %')
    
    if (fluct_pct < 1.):
        video_OK = True
    else:
        print("Frame period fluctuation unusually large. Check your video!")
        
        
    return omexmlstr, timestamps, video_OK



