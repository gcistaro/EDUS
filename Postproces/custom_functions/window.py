import numpy as np

def CutWindow(freq_eV, Ew_norm, limits=None):
    # use boolean masks (faster than np.where)
    if limits is None:
        thr = 0.05 * Ew_norm.max()     # max() is faster than np.max()
        mask = (Ew_norm >= thr) & (freq_eV > 0)
    else:
        low, high = limits
        mask = (freq_eV > low) & (freq_eV < high)

    # return indices where mask=True
    return np.nonzero(mask)[0]
