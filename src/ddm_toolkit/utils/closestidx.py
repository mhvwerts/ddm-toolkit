import numpy as np

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
