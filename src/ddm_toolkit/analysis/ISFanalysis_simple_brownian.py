import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ddm_toolkit.utils import closestidx, conf95
from .ISF_analysis_simple_brownian_result import ISFanalysis_simple_brownian_result

def ISFanalysis_simple_brownian(IA, D_guess, refine_guess = True,
                                verbose = False):
    """Analyze an ISF using the simple Brownian model.

    The fitting program only needs an educated guess for the expected
    diffusion coefficient, and automatically adapts to best fitting
    practices. The simple Brownian model implies that the ISF is radially
    symmetric (note: the analysis does not check that it it is!). After
    calulation of the radially averaged ISF, the ISF(tau) is fit over a
    reasonable range of q values (see the explanation in our
    forthcoming paper), limited by objective criteria of the
    fittability of the exponential curves.

    Parameters
    ----------
    IA : ImageStructureFunction
        The calculated image structure function (ImageStructureFunction
        object).
    D_guess : float
        Educated guess for the diffusion coefficient. This will be used
        as an initial guess for setting fit ranges etc. The user-supplied
        guess will be refined by the program by successive fits.
        The refined guess will be used for the full fitting procedure
        which factorizes the radialized ISF into A(q) B(q) and f(k(q), tau).
    refine_guess : bool, optional (default: True)
        If 'False', the program will not refine the user-supplied guess,
        but use it directly for setting the fit ranges. This is NOT
        recommended, but may be useful in case one is analyzing a
        reference sample for which the diffusion coefficient is known.
    verbose : bool, optional (default: False)
        If 'True', print analysis results and diagnostic info to console.

    Returns
    -------
    dict
        Object with detailed analysis results and reporting methods.


    """
    assert IA.real_world_units, "Real-world units should already have been applied to the ISF"

    # Return result data will be stored in ISFanalysis_simple_brownian_result object
    result = ISFanalysis_simple_brownian_result()
    result.D_guess = D_guess
    result.tau = IA.tau
    result.q = IA.q

    # Factors for setting the "reasonably fittable range" of q
    # These factors might be put in parameter file, but hard-coded for now.
    # Were found to be reasonable by visual inspection and exponential
    # fitting experience.
    # We made 'safe' choices.
    optfac = 8.0 # optimal full data window 8 times the time constant
    highfac = 3.0 # shortest time constant (highest k) 3 times shortest delta tau
    lowfac = 2.0 # longest time constant (lowest k) half the full ISF data window

    if IA.isRadialAverage:
        IAqtau = IA.ISFqtau
    else:
        # radial averager
        IAqtau = np.zeros((len(IA.tauf),len(IA.u)))
        for i in range(len(IA.tauf)):
            IAqtau[i,:] = IA.radavg(i)
        # remove "zero-frequency garbage" at center
        IAqtau[:,0] = 0.0

    # store in result dictionary
    # for external plotting
    result.radISF_q = IA.q
    result.radISF_tau = IA.tau
    result.radISF_qtau = IAqtau[:,:]


    # ===============================================
    # Procedure for fitting simple Brownian DDM model
    # ===============================================

    # model function: Brownian ISF function (tau version) at a given q
    def ISFmod(tau, A, B, k):
        f = np.exp(-k*tau)
        return A*(1-f) + B


    # model function: k = Dq^2
    def kmod(q, D):
        return D*q**2


    result.D_guess_user = D_guess
    if verbose:
        print('D_guess (user):',D_guess,'µm2/s')

    # step 1: select 'optimal' q for first ISF(tau) fit
    #
    # We consider an optimal rate constant for fitting to be k_opt = 8/tau_max
    # where tau_max is the longest  accessible time lag from the measurement
    # i.e. the corresponding time constant is 1/8th of the total tau window
    tau_max = IA.tau[-1]
    deltau = IA.tau[1] - IA.tau[0]

    # derive limits on reliable k from \tau_max and \Delta \tau
    k_opt = optfac / tau_max
    k_high = 1.0 / (highfac * deltau)
    k_low = lowfac / tau_max

    if refine_guess:
        # steps 2...5: obtain improved D_guess, by iterative fitting of ISF at single q
        #
        # epsconv, maxiter may be made parametrable
        epsconv = 0.1
        maxiter = 1000
        ii = 0
        doloop = True
        while(doloop):
            # find the q value where the time constant is exlected to be 'optimal'
            # for the data set, given D_guess
            q_opt = np.sqrt(k_opt/D_guess)
            iq_opt = closestidx(IA.q, q_opt)
            q_opt = IA.q[iq_opt]

            # fit ISF at select q values
            # create initial guess and parameter set-up

            ## scipy.optimize.curve_fit based fitting
            kinit = k_opt
            Binit = IAqtau[0, iq_opt]
            Ainit = IAqtau[-1, iq_opt] - Binit
            p_guess = [Ainit, Binit, kinit]
            p_fit, p_fitcov = curve_fit(ISFmod, IA.tau[1:], IAqtau[1:,iq_opt],
                                        p0 = p_guess)
            k_q = p_fit[2]
            A_q = p_fit[0]
            B_q = p_fit[1]

            # refined guess
            D_guess2 = k_q / (q_opt**2)
            eps = np.abs(D_guess-D_guess2)/D_guess
            if (eps<epsconv):
                doloop = False
            ii += 1
            if(ii>maxiter):
                print('WARNING: maximum number of iterations reached in')
                print('         refinement of D_guess.')
                print('         eps = ',eps)
                print('         D_guess =',D_guess)
                print('         D_guess2 =',D_guess2)
                print('         (probably infinitely repetitive iterations...)')
                doloop = False
                D_guess = (D_guess + D_guess2)/2
            else:
                D_guess = D_guess2

    result.D_guess_refined = D_guess
    if verbose:
        print('D_guess (refined):', D_guess, 'µm2/s')

    # Step 6: on basis of the new D_guess, establish a window of q_s
    # where reasonable fit quality is expected
    # this window contains 2 regions: below k_opt and beyond k_opt
    # first find these k_low, k_high
    q_opt = np.sqrt(k_opt/D_guess)
    iq_opt = closestidx(IA.q, q_opt)
    q_opt = IA.q[iq_opt]

    q_low = np.sqrt(k_low/D_guess)
    iq_low = closestidx(IA.q, q_low)
    q_low = IA.q[iq_low]

    q_high = np.sqrt(k_high/D_guess)
    iq_high = closestidx(IA.q, q_high)
    q_high = IA.q[iq_high]

    result.q_opt = q_opt
    result.iq_opt = iq_opt
    result.q_low = q_low
    result.iq_low = iq_low
    result.q_high = q_high
    result.iq_high = iq_high

    if verbose:
        print('q_low: ',q_low,'µm-1')
        print('q_opt: ',q_opt,'µm-1')
        print('q_high:',q_high,'µm-1')
        print('q_max: ',IA.q[-1],'µm-1')

    assert iq_low < iq_opt < iq_high, "unexpected index ordering"

    k_q = np.zeros_like(IA.q)
    A_q = np.zeros_like(IA.q)
    B_q = np.zeros_like(IA.q)
    #
    # FIT LOOP / PART A: Fit between q_low and q_opt
    #
    # here: fit full data (since optimal fit window would be larger than
    #       available data)
    for iqf in range(iq_low, iq_opt+1):
        ## scipy.optimize.curve_fit based fitting
        qf = IA.q[iqf]
        kinit = D_guess*qf**2
        Binit = IAqtau[0, iqf]
        Ainit = IAqtau[-1, iqf] - Binit
        p_guess = [Ainit, Binit, kinit]
        p_fit, p_fitcov = curve_fit(ISFmod, IA.tau[1:], IAqtau[1:,iqf],
                                    p0 = p_guess)
        k_q[iqf] = p_fit[2]
        A_q[iqf] = p_fit[0]
        B_q[iqf] = p_fit[1]

        # sample specific fits (for plotting and diagnostic purposes)
        if (iqf==iq_low):
            result.ISFmodelfit_qlow = ISFmod(IA.tau, *p_fit)

        if (iqf==iq_opt):
            result.ISFmodelfit_qopt = ISFmod(IA.tau, *p_fit)
    #
    # FIT LOOP / PART B: Fit between q_opt and q_high
    #
    # here: reduce fit range to have 'optimal fit window'
    # use fit result for new initial guess
    for iqf in range(iq_opt+1, iq_high+1):
        ## scipy.optimize.curve_fit based fitting
        qf = IA.q[iqf]
        kinit = D_guess*qf**2 # take expected value on basis of D_guess
        Binit = B_q[iqf-1] # from previous fit
        Ainit = A_q[iqf-1] # take amplitude guess from previous fit
        p_guess = [Ainit, Binit, kinit]
        # reduce fit range to have 'optimal fit window'
        tau_opt = optfac/kinit
        itau_opt = closestidx(IA.tau, tau_opt)
        tau_opt = IA.tau[itau_opt]
        # fit
        try:
            p_fit, p_fitcov = curve_fit(ISFmod,
                                        IA.tau[1:itau_opt], IAqtau[1:itau_opt,iqf],
                                        p0 = p_guess)
        except RuntimeError:
            # catching rare fit error
            # The fit does not converge.
            # This usually happens for points where noise is significant,
            # and the amplitude A(q) is already small.
            # One way to detect such situation may be to look at B(q)/A(q)
            # since A(q) is usually monotonously descending,
            # we may use this situation to simply stop fitting
            # and set iq_high to the last detected point
            print('WARNING: optimal fit not found at index {0:d} (rel. ix = {1:d})'.format(iqf, iqf-(iq_opt+1)))

            ## ONE WAY OF HANDLING THIS SITUATION:
            ## (in this case we can monitor which points have problems)
            ## since the loop goes all the way
            # print('         copying previous p_fit (TODO: FIX THIS)')
            # # TODO: THIS SHOULD BE REPLACED BY SETTING THESE TO NaN
            # # and treat them as missing points in the final fit
            # p_fit[2] = k_q[iqf-1]
            # p_fit[0] = A_q[iqf-1]
            # p_fit[1] = B_q[iqf-1]

            ## OTHER WAY OF HANDLING THIS SITUATION:

            print('         stopping the analysis here. updating iq_high to reflect last valid point')
            iq_high = iqf - 1
            break
        ## B(q)/A(q) > limit stopping criterion
        ## (i.e. noise/signal ratio) "noise-to-signal ratio exceeded... stopping analysis"
        ## TODO: make this configurable, remove magic number 25.
        if (abs(p_fit[1]/p_fit[0]) > 25.):
            print('WARNING: B-to-A ratio exceeded before iq_high.')
            print('         stopping the analysis here. updating iq_high to reflect last valid point')
            iq_high = iqf - 1
            break
        k_q[iqf] = p_fit[2]
        A_q[iqf] = p_fit[0]
        B_q[iqf] = p_fit[1]

    # sample fit for iq_high (for plotting and diagnostic purposes)
    p_fit[2] = k_q[iq_high]
    p_fit[0] = A_q[iq_high]
    p_fit[1] = B_q[iq_high]
    result.ISFmodelfit_qhigh = ISFmod(IA.tau, *p_fit)


    # store A(q), B(q), k(q)
    result.k_q = k_q
    result.A_q = A_q
    result.B_q = B_q

    # Steps 7-8: fit k

    ####
    q = result.q

    p_guess = [D_guess]
    p_fit, p_fitcov = curve_fit(kmod,
                                q[iq_low:iq_high], k_q[iq_low:iq_high],
                                p0 = p_guess)
    Dfit = p_fit[0]
    Ndata = len(k_q[iq_low:iq_high])
    Dstdv = p_fitcov[0,0]**0.5
    Dci = conf95(Dstdv, Ndata, 1)

    if verbose:
        print('D (fit):',Dfit,'µm2/s (+/- ', Dci,', 95% CI)')

    result.D_fit = Dfit
    result.D_fit_CI95 = Dci

    # re_update q_high (in case analysis finished prematurely)
    q_high = IA.q[iq_high]
    result.q_high = q_high
    result.iq_high = iq_high

    return result
