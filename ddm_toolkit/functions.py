# -*- coding: utf-8 -*-
"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

functions.py:
    additional functions
"""
import numpy as np
from scipy.stats.distributions import t as student_t


# GENERAL utility functions

def closestidx(vec, val):
    """Find the index of the vector element closest to a value.
    
    Parameters
    ----------
    vec : 1D numpy vector
        Vector in which the value is searched.
    val : float
        Value to be found in vector.
        
    Returns
    -------
    int
        Index of element in `vec` with value closest to `value`
    """
    dist=abs(vec-val)
    return np.argmin(dist)



def conf95(stdev, Ndata, Npar):
    """Calculate (one half) of the symmetric 95% confidence interval.
    
    The symmetric 95% confidence interval is calculated from the standard
    deviation and number of degrees of freedom, using Student's t
    distribution.
    
    Parameters
    ----------
    stdev : float
        Standard deviation.
    Ndata : int
        Number of data points.
    Npar : int
        Number of parameters.
    
    Returns
    -------
    float
        The half-width of the 95% confidence interval, such that it
        can be reported in the tradtional +/- manner.
    
    based on:
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
    """
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)
    dof = max(0, Ndata - Npar)
    tval = student_t.ppf(1.0-alpha/2., dof)
    return (stdev*tval)


