import numpy as np
import cv2 as cv
import math


class pyr:
    @staticmethod
    def cvResizeBicubic(X, factor=2):
        halfSize = X.shape[1] // factor, X.shape[0] // factor
        return cv.resize(X, dsize=halfSize)

    @staticmethod
    def custom(F):
        return F


def dirac(*dim):
    C = np.zeros(dim, dtype=np.uint)
    center = tuple(d // 2 for d in dim)
    C[center] = 1
    return C


def box(*dim):
    C = np.ones(dim, dtype=np.uint)
    k = 1 / C.size
    return C * k


def binomial(*dim):
    c = np.empty(dim[0], dtype=np.uint)
    n = dim[0] - 1

    for i in range(dim[0]):
        c[i] = math.factorial(n)/(math.factorial(i)*math.factorial(n - i))

    if len(dim) > 1:
        c = c.reshape(dim[0], 1)
        C = c @ c.T
    else:
        C = c

    k = 1 / np.sum(C)
    return C * k


def median(self, X, W):
    X_ = np.repeat(X.flatten(), W.flatten())
    X_ = np.sort(X_)
    return (X_[(X_.size - 1) // 2] + X_[X_.size // 2]) / 2



def conv2d(X, F, pad='constant'):
    ri, rj = (F.shape[0] - 1) // 2, (F.shape[1] - 1) // 2
    X_ = np.pad(X, ((ri, ri), (rj, rj)), mode=pad)
    Y = np.zeros_like(X)
    f = F.flatten()

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] = (X_[i:i + 2*ri + 1, j:j + 2*rj + 1].flatten() @ f)

    return X_
