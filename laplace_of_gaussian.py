import scipy as sp
import numpy as np
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
from skimage import data


def LoGFilter(img):
    LoG = gaussian_laplace(img, 2)
    thresh = np.absolute(LoG).mean() * 0.75
    output = np.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y-1:y+2, x-1:x+2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p.any()):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thresh) and zeroCross:
                output[y, x] = 1
    return output
