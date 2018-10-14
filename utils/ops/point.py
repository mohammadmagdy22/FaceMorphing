import numpy as np


def normalize(X):
    Xmax = np.iinfo(X.dtype).max
    return X.astype(np.double) / Xmax


def histo(X, bins, Xmax, cumulative=False):
    H = np.zeros(bins, np.uint)
    Xmax = (Xmax or np.iinfo(X.dtype).max) + 1

    for x in np.nditer(X):
        i = (x * bins // Xmax).astype(np.int)
        H[i] += 1

    if cumulative:
        H = np.array(
            [np.sum(H[:i+1]) for i in range(bins)]
        )

    return H


def contrastBrightness(X, c, b):
    maxVal = np.iinfo(X.dtype).max
    return np.max(0, np.min(c * X.copy() + b, maxVal))


def invert(X):
    maxVal = np.iinfo(X.dtype).max
    return maxVal - X


def threshold(X, threshold, x0, x1):
    return np.where(X < threshold, x0, x1)


def autocontrast(X, q=0, minmax=None):
    xMin, xMax = minmax or (0, np.iinfo(X.dtype).max)

    if q > 0:
        h = histo(X, bins=256, cumulative=True)
        xLow = np.min(np.argwhere(h >= X.size * q))
        xHigh = np.max(np.argwhere(h <= X.size * (1 - q)))
    else:
        xLow, xHigh = X.min(), X.max()

    return xMin + (X - xLow) * ((xMax - xMin)/(xHigh - xLow))











