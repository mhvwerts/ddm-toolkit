#!/usr/bin/env python3
# coding: utf-8
#
# simulate Brownian motion, generate synthetic video frames,
# analyze with DDM algorithm
#
# The DDM Team, 2020-2021
#
# diffusion coefficient in => diffusion coefficient out
#
# STEP 7: Show a single result of a 'simul6_multicycle' run
#
# Command line usage:
#     python simul7_plot_multicycle.py datafiles/x3dc_00001.pkl 
#
# Jupyter Notebook/Lab usage:
#     from simul7_plot_multicycle import plot_single_cycle_result
#     plot_single_cycle_result('datafiles/x3dc_00001.pkl')
#
# Can also be run in Spyder (development/testing)
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt



def plot_single_cycle_result(fpname):
    """
    Read and display the results from a single simulation-analysis cycle.
    
    This reads and plots the results from a single simulation-analysis cycle
    in a multicycle run.
    
    Currently, a very rudimentary version for diagnostic purposes and for
    guiding further development of multicycle analysis tools.
    
    The multicycle run is for heavy testing, and also for obtaining statistics,
    and parametric studies.

    Parameters
    ----------
    fpname : string
        Pathname of the file.

    Returns
    -------
    None.

    """
    
    print('=========================')
    print(' File: '+fpname)
    print('=========================')
    print()
    with open(fpname, 'rb') as f1:
        sim = pickle.load(f1)
        res = pickle.load(f1)
        
    
    print('TODO: print simulation parameters used')
    print()
    
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
    q = res['q']
    
    
    # ===================================
    # OUTPUT RESULTS
    # ===================================
    #
    # print results (currently same output as 'verbose = True')
    print('-------------------------')
    print('Brownian analysis summary')
    print('-------------------------')
    print('D_guess (user):',sim.D_guess,'µm2/s')
    print('D_guess (refined):', D_guess_refined, 'µm2/s')
    print('q_low: ',q_low,'µm-1')
    print('q_opt: ',q_opt,'µm-1')
    print('q_high:',q_high,'µm-1')
    #print('q_max: ',IA.q[-1],'µm-1')
    print('D (fit):', D_fit, 'µm2/s (+/- ', D_fit_ci95, ', 95% CI)')
    print()
    
    
    
    # # PLOT radially averaged ISF
    # plt.figure("ISF (radially averaged)")
    # plt.clf()
    # plt.subplot(211)
    # plt.title('radially averaged ISF')
    # for itau in range(0,10):
    #     plt.plot(IA.q,IAqtau[itau,:])
    # for itau in range(20,100,10):
    #     plt.plot(IA.q,IAqtau[itau,:])
    # plt.xlabel('q [µm-1]')
    # plt.ylabel('ISF')
    # plt.subplot(212)
    # plt.imshow(IAqtau.T, origin = 'lower', aspect ='auto',
    #            extent =(IA.tau[0],IA.tau[-1],
    #                     IA.q[0],IA.q[-1]))
    # plt.colorbar()
    # plt.ylabel('q [µm-1]')
    # plt.xlabel('tau [s]')
    
    
    # # PLOT explicitly some fits (diagnostic to verify if fits OK)
    # plt.figure('Fits at q_low, q_opt, q_high')
    # plt.clf()
    # plt.subplot(311)
    # plt.title('Fits at q_low, q_opt, q_high (refined D_guess)')
    # plt.plot(IA.tau[1:], IAqtau[1:,iq_low],'o')
    # plt.plot(IA.tau, res['ISFmodelfit_qlow'])
    # plt.ylabel('ISF')
    # plt.subplot(312)
    # plt.plot(IA.tau[1:], IAqtau[1:,iq_opt],'o')
    # plt.plot(IA.tau, res['ISFmodelfit_qopt'])
    # plt.ylabel('ISF')
    # plt.subplot(313)
    # plt.plot(IA.tau[1:], IAqtau[1:,iq_high],'o')
    # plt.plot(IA.tau, res['ISFmodelfit_qhigh'])
    # plt.ylabel('ISF')
    # plt.xlabel('tau [s]')
    
    
    # PLOT final result: A(q), B(q), k(q) and fit of k(q)
    plt.figure('Result of simple Brownian analysis of DDM')
    plt.clf()
    plt.subplot(311)
    plt.title(fpname)
    plt.plot(q[iq_low:iq_high],k_q[iq_low:iq_high],'o')
    plt.plot(q,sim.D*q**2, label = 'expected')
    plt.plot(q,D_fit*q**2, label = 'fit')
    plt.ylim(0, np.max(k_q)*1.3)
    plt.xlim(xmin=0,xmax=q[-1])
    plt.ylabel('k(q) [s-1]')
    plt.legend()#TO DO: make file selector

    plt.subplot(312)
    plt.plot(q[iq_low:iq_high],A_q[iq_low:iq_high],'o')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0,xmax=q[-1])
    plt.ylabel('A(q) [a.u.]')
    plt.subplot(313)
    plt.plot(q[iq_low:iq_high],
             B_q[iq_low:iq_high]/A_q[iq_low:iq_high],'o')
    plt.xlim(xmin=0,xmax=q[-1])
    plt.ylabel('B(q)/A(q)')
    plt.xlabel('q [µm-1]')


if __name__ == '__main__':
    fpname = './datafiles/xab2_00001.pkl'
    if len(sys.argv)==2:
        fpname=sys.argv[1]
    
    plot_single_cycle_result(fpname)
    print('(close all open child windows to terminate script)')
    plt.show()
