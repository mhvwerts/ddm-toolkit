#!/usr/bin/env python3
# coding: utf-8
#
#
# Analysis of Image Structure Functions in DDM
#
# This analysis is both for the 'simul' and 'video' workflows.
#
# The DDM Team, 2020-2021
#
#
# xDDM1 : Inspect ImageStructureFunction (videofig player)
#

#%% Insert path to import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

#%%
import numpy as np
from matplotlib import pyplot as plt

#%%
from ddm_toolkit.ddm import ImageStructureFunction

#%%
from ddm_toolkit.params  import DDMParams_from_configfile_or_defaultpars

#%%
from ddm_toolkit.videofig import videofig 

#%%
if (__name__ == '__main__'):

    # Get configuration parameters (default is default simulation)
    params = DDMParams_from_configfile_or_defaultpars()
    
    # ISF file
    ISF_fpn = params.ISF_outfpn
    
    # is source file a full ISF or a radially average ISF?
    assert not params.ISF_radialaverage, "inspect ISF only possible for full ISF (this ISF is radially averaged)"
    
    # overdrive parameter (boost ISF display brightness)
    img_overdrive = params.ISF_viewer_overdrive
    
    
    # load image structure function
    print('loading file: ',ISF_fpn)
    IA = ImageStructureFunction.fromFile(ISF_fpn)
    
    Ni = IA.ISF.shape[0] 
        
    vmx = IA.ISF.max() / img_overdrive # avoid autoscale of colormap
    
    vf_redraw_init=False
    vf_redraw_img=None
    def vf_redraw(fri, ax):
        global ISFob
        global vmx
        global vf_redraw_init
        global vf_redraw_img
        global k_max_over_2
        if not vf_redraw_init:
            vf_redraw_img=ax.imshow(IA.ISF[fri], 
                                    vmin = 0.0, vmax = vmx, 
                                    origin = 'lower', animated = True)

            vf_redraw_init=True
        else:
            vf_redraw_img.set_array(IA.ISF[fri])
    
    print("[ENTER]: toggle pause/play")
    print("[LEFT]/[RIGHT]: scroll frames")
    print("[MOUSE]: manipulate time bar")
    
    
    videofig(Ni, vf_redraw, play_fps=10)
    
    


