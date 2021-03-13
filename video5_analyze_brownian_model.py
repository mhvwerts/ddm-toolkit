#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# by Greshma Babu and Martinus Werts, 2020
#
# diffusion coefficient in => diffusion coefficient out
#
# STEP 5: Analysis of simulation result
#
from sys import argv
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit import ImageStructureFunction
from ddm_toolkit import ISFanalysis_simple_brownian



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


ISF_fpn = params['ISFengine']['ISF_fpn']

# real-world dimensions
um_p_pix = float(params['realworld']['px_size'])
s_p_frame = float(params['realworld']['frm_period'])

D_guess = float(params['analysis_brownian']['D_guess'])


#%%
# =========================================
# LOAD and PREPARE Image Structure Function
# =========================================
#
# load image structure function & apply REAL WORLD UNITS!
IA = ImageStructureFunction.fromfilename(ISF_fpn)
IA.real_world(um_p_pix, s_p_frame)


# ===================================
# FIT THE BROWNIAN MODEL
# ===================================
#
# INPUT: D_guess => FROM PARAMETER FILE


# perform analysis of the ISF using the simple Brownian model
res = ISFanalysis_simple_brownian(IA, D_guess)

# get results of analysis
IAqtau = res['radISF_qtau']
iq_low = res['iq_low']
q_low = res['q_low']
iq_opt = res['iq_opt']
q_opt = res['q_opt']
iq_high = res['iq_high']
q_high = res['q_high']
D_guess_refined = res['D_guess_refined']
k_q = res['k_q']
A_q = res['A_q']
B_q = res['B_q']
D_fit = res['D_fit']
D_fit_ci95 = res['D_fit_CI95']



# ===================================
# OUTPUT RESULTS
# ===================================
#
# print results (currently same output as 'verbose = True')
print('D_guess (user):',D_guess,'µm2/s')
print('D_guess (refined):', D_guess_refined, 'µm2/s')
print('q_low: ',q_low,'µm-1')
print('q_opt: ',q_opt,'µm-1')
print('q_high:',q_high,'µm-1')
print('q_max: ',IA.q[-1],'µm-1')
print('D (fit):', D_fit, 'µm2/s (+/- ', D_fit_ci95, ', 95% CI)')


# PLOT radially averaged ISF
plt.figure("ISF (radially averaged)")
plt.clf()
plt.subplot(211)
plt.title('radially averaged ISF')
for itau in range(0,10):
    plt.plot(IA.q,IAqtau[itau,:])
for itau in range(20,len(IA.tau),10):
    plt.plot(IA.q,IAqtau[itau,:])
plt.xlabel('q [µm-1]')
plt.ylabel('ISF')
plt.subplot(212)
plt.imshow(IAqtau.T, origin = 'lower', aspect ='auto',
           extent =(IA.tau[0],IA.tau[-1],
                    IA.q[0],IA.q[-1]))
plt.colorbar()
plt.ylabel('q [µm-1]')
plt.xlabel('tau [s]')


# PLOT explicitly some fits (diagnostic to verify if fits OK)
plt.figure('Fits at q_low, q_opt, q_high')
plt.clf()
plt.subplot(311)
plt.title('Fits at q_low, q_opt, q_high (refined D_guess)')
plt.plot(IA.tau[1:], IAqtau[1:,iq_low],'o')
plt.plot(IA.tau, res['ISFmodelfit_qlow'])
plt.ylabel('ISF')
plt.subplot(312)
plt.plot(IA.tau[1:], IAqtau[1:,iq_opt],'o')
plt.plot(IA.tau, res['ISFmodelfit_qopt'])
plt.ylabel('ISF')
plt.subplot(313)
plt.plot(IA.tau[1:], IAqtau[1:,iq_high],'o')
plt.plot(IA.tau, res['ISFmodelfit_qhigh'])
plt.ylabel('ISF')
plt.xlabel('tau [s]')


# PLOT final result: A(q), B(q), k(q) and fit of k(q)
plt.figure('Result of simple Brownian analysis of DDM')
plt.clf()
plt.subplot(311)
plt.title('Simple Brownian model analysis of ISF')
plt.plot(IA.q[iq_low:iq_high],k_q[iq_low:iq_high],'o')
plt.plot(IA.q,D_guess*IA.q**2, label = 'init. guess')
plt.plot(IA.q,D_fit*IA.q**2, label = 'fit')
plt.ylim(0, np.max(k_q)*1.3)
plt.xlim(xmin=0,xmax=IA.q[-1])
plt.ylabel('k(q) [s-1]')
plt.legend()
plt.subplot(312)
plt.plot(IA.q[iq_low:iq_high],A_q[iq_low:iq_high],'o')
plt.ylim(ymin=0)
plt.xlim(xmin=0,xmax=IA.q[-1])
plt.ylabel('A(q) [a.u.]')
plt.subplot(313)
plt.plot(IA.q[iq_low:iq_high],
         B_q[iq_low:iq_high]/A_q[iq_low:iq_high],'o')
plt.xlim(xmin=0,xmax=IA.q[-1])
plt.ylabel('B(q)/A(q)')
plt.xlabel('q [µm-1]')

#TODO make this part conditional, do not display if running in Spyder
print('(close all open child windows to terminate script)')
plt.show()
