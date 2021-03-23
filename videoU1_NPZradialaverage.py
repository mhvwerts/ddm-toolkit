#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:44:34 2021

@author: werts
"""

from sys import argv
import os.path
from configparser import ConfigParser


from ddm_toolkit import ImageStructureFunction



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
    raise Exception('invalid number of arguments')
 
params = ConfigParser(interpolation=None)
params.read(argfn)

fnbase, fnext = os.path.splitext(argfn)


# is source file a full ISF or a radially average ISF?
isISFradialaverage = False
try:
    if params['ISFengine']['ISF_radialaverage']=='True':
        isISFradialaverage = True
except KeyError:
    pass


#%%
# =========================================
# LOAD and PREPARE Image Structure Function
# =========================================
#
# load image structure function & apply REAL WORLD UNITS!

# break point: TO DO!


if isISFradialaverage:
    print('ISF NPZ file is already a radial average!')
else:
    ISF_fpn = fnbase+'_ISF.npz'
    IA = ImageStructureFunction.fromFile(ISF_fpn)
    ISFRadAvg_fpn = fnbase+'_ISFRadAvg.npz'
    IA.saveRadAvg(ISFRadAvg_fpn)


    


