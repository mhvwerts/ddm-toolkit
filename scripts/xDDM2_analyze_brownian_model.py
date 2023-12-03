#!/usr/bin/env python3
# coding: utf-8
#
# Analysis of Image Structure Functions in DDM
#
# This analysis is both for the 'simul' and 'video' workflows.
#
# The DDM Team, 2020-2021
#
#
# xDDM2 : Analyze ISF in terms of Simple Brownian Model
#
#

#%% Insert path to import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

#%%
from matplotlib import pyplot as plt

#%%
from ddm_toolkit.params import DDMParams_from_configfile_or_defaultpars

#%%
from ddm_toolkit.ddm import ImageStructureFunction, ImageStructureFunctionRadAvg
from ddm_toolkit.analysis import ISFanalysis_simple_brownian




# ===================================
# READ SIMULATION/ANALYSIS PARAMETERS
# ===================================
params = DDMParams_from_configfile_or_defaultpars()

# ISF file
ISF_fpn = params.ISF_outfpn


# =========================================
# LOAD and PREPARE Image Structure Function
# =========================================
#
# load image structure function

if params.ISF_radialaverage:
    IA = ImageStructureFunctionRadAvg.fromFile(ISF_fpn)
else:
    IA = ImageStructureFunction.fromFile(ISF_fpn)
    
assert IA.real_world_units, "Real-world units should already have been applied"



# ===================================
# FIT THE BROWNIAN MODEL
# ===================================
#
# Initial guess (from parameter file)
D_guess = params.D_guess

# perform analysis of the ISF using the simple Brownian model
# the results are in 'res' (object)
res = ISFanalysis_simple_brownian(IA, D_guess)


# ===================================
# OUTPUT RESULTS
# ===================================
#
res.print_report()

# PLOT radially averaged ISF
res.show_ISF_radavg()

# PLOT explicitly some fits (diagnostic to verify if fits OK)
res.show_fits()

# PLOT final result: A(q), B(q), k(q) and fit of k(q)
res.show_Aq_Bq_kq()


print('(close all open child windows to terminate script)')
plt.show()
