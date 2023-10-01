import numpy as np


def wthresh(A: np.ndarray, thresh: float) -> np.ndarray:
    """Port of matlabs hard thresholding wthresh.
    see: https://www.mathworks.com/help/wavelet/ref/wthresh.html

    Args:
        A (array): array on which to apply the thresholding.
        threshold (float): threshold value.

    Returns:
        array: thresholded array.
    """
    out = A.copy()
    out[np.abs(out) < thresh] = 0
    return out
