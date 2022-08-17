#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# STEP 2: Visualize synthetic video (videofig player)
#
# The DDM Team, 2020-2021
#
# Usage:
#        python3 simul2_inspect_video.py [<name of parameter file>]
#
#%% Insert path to import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

#%%
from ddm_toolkit.params import DDMParams_from_configfile_or_defaultpars

#%%
from ddm_toolkit.workflows import simul2_load_simulation_result_file

#%%
from ddm_toolkit.videofig import videofig 

#%%
if __name__ == "__main__":
    
    # Get simulation/analysis parameters
    params = DDMParams_from_configfile_or_defaultpars()
    simulfpn = params.vidfpn

    # LOAD data (and parameters stored inside the file, but these are not used)
    ims, params_simulfile = simul2_load_simulation_result_file(simulfpn)
    
    # prepare videoplayer
    if (params.video_Nview < 0):
        Ni = ims.shape[0]
    else:
        Ni = params.Nview
        
    vmx = ims.max() / params.video_overdrive # avoid autoscale of colormap
    
    vf_redraw_init = False
    vf_redraw_img = None
    
    def vf_redraw(fri, ax):
        global ims
        global vmx
        global vf_redraw_init
        global vf_redraw_img
        if not vf_redraw_init:
            vf_redraw_img=ax.imshow(ims[fri], 
                                    vmin = 0.0, vmax = vmx, 
                                    origin = 'lower', animated = True)
            vf_redraw_init=True
        else:
            vf_redraw_img.set_array(ims[fri])
    
    print("[ENTER]: toggle pause/play")
    print("[LEFT]/[RIGHT]: scroll frames")
    print("[MOUSE]: manipulate time bar")
    
    videofig(Ni, vf_redraw, play_fps=10)

