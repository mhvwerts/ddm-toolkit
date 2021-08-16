#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze experimental video


STEP 3: Read and calculate ISF of experimental video



This script uses 'tifffile' by C. Gohlke
More documentation on this library:
    https://github.com/cgohlke/tifffile
(especially in the "Examples" section of 'README.rst')




python video3_ISF_from_video.py <filename.txt>
    this will use a configuration file, which enables us
    to set a ROI window, a start and stop frame etc.


"""

from sys import argv
import os.path
from configparser import ConfigParser

from ddm_toolkit.tifftools import TiffFile, tifffile_version
from ddm_toolkit import tqdm
from ddm_toolkit import ImageStructureEngineSelector
from ddm_toolkit import ImageStructureFunction

print('tifffile version: ', tifffile_version)


#%% 
# ==============================
# get PROCESSING PARAMETERS
# ==============================
# Read parameter file, default to "video0_test_params.txt"
# if nothing (easier with Spyder)

argc = len(argv)
if argc == 1:
    argfn = "video0_test_params.txt"
elif argc == 2:
    argfn = argv[1]
else:
    raise Exception('invalid number of arguments.')
    
params = ConfigParser(interpolation=None)
params.read(argfn)

fnbase, fnext = os.path.splitext(argfn)

infp = params['videofile']['pathname']
frm_start = int(params['videofile']['frm_start'])
frm_end = int(params['videofile']['frm_end'])
ROI_x1 = int(params['videofile']['ROI_x'])
ROI_y1 = int(params['videofile']['ROI_y'])
ROI_size = int(params['videofile']['ROI_size'])
frm_Npreview = int(params['videofile']['frm_Npreview'])
vid_overdrive = float(params['videofile']['display_overdrive'])
ROIcontrast = float(params['videofile']['display_ROIcontrast'])
ROI_x2 = ROI_x1 + ROI_size
ROI_y2 = ROI_y1 + ROI_size

ISE_Nbuf = int(params['ISFengine']['ISE_Nbuf'])
ISE_Npx = ROI_size
doISFradialaverage = False
try:
    if params['ISFengine']['ISF_radialaverage']=='True':
        doISFradialaverage = True
except KeyError:
    pass

try:
    ISE_type = int(params['ISFengine']['ISE_type'])
except KeyError:
    ISE_type = 0    
ImageStructureEngine = ImageStructureEngineSelector(ISE_type)


#%% load video and calculate ISF
print('file being processed: ', infp)
tif = TiffFile(infp)

tiflen = len(tif.pages) - 1 # remove last frame (sometime causes problems)
print('TIFF length: {0:d} frames'.format(tiflen))

# first image to get dimensions
imgfull = tif.pages[0].asarray()
# enable full image processing
if ROI_size < 0:
    ROI_x2 = imgfull.shape[1]
    ROI_y2 = imgfull.shape[0]

print('ROI: x={0:d}...{1:d}  y={2:d}...{3:d}'.format(ROI_x1,
                                                     ROI_x2,
                                                     ROI_y1,
                                                     ROI_y2))
if (frm_end > tiflen) or (frm_end < 1):
    frm_end = tiflen

print('selected frames: start={0:d}  end={1:d}'.format(frm_start,
                                                       frm_end))
Nframes = frm_end - frm_start

#TODO: further configure engine
#      in particular apodization 
ISE = ImageStructureEngine(ISE_Npx, ISE_Nbuf)

# loop over selected images in file
# tqdm just makes a progress bar (useful for large files!)
for i in tqdm(range(Nframes)):
    imgfull = tif.pages[i + frm_start].asarray()
    # get ROI only
    img = imgfull[ROI_y1:ROI_y2,ROI_x1:ROI_x2]
    ISE.push(img)
        
tif.close()

print()
print('#frames contributing to averaged ISF (ISFcount): {0:d}'.format(ISE.ISFcount))



print('No ISF file saved!')
assert 1==0, 'TO DO! resultfn (ISF file name) to be read from video_params. txt PARAMETER FILE and used (for compatibility with new xDDM1 and xDDM2)'




if doISFradialaverage:
    # radially average the ISF and save it
    IA = ImageStructureFunction.fromImageStructureEngine(ISE)
    # REMOVE: resultfn = fnbase+'_ISFRadAvg.npz'
    IA.saveRadAvg(resultfn)
else:
    # just save the whole x,y ISF
    # REMOVE: resultfn = fnbase+'_ISF.npz'
    ISE.save(resultfn)

    

