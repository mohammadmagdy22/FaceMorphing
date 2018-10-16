import matplotlib.pyplot as plt
import os
import json
import numpy as np
import cv2 as cv
import scipy.spatial
from scipy import interpolate as interp
from utils import ops


def triangulate(vtxA, vtxB, k):
    vtxM = k * vtxA + (1 - k) * vtxB
    d = scipy.spatial.Delaunay(vtxM, qhull_options='QJ')
    return d


class CorrespondenceUI:
    def __init__(self, A, B, save=True):
        self.ax, self.info, self.save = None, {}, save
        self.featuresOk = False
        self.fig = plt.figure(figsize=(10, 5))
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1)
        self.fig.canvas.mpl_connect('axes_enter_event', self.onEnterAxis)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)

        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(A * 0.6, interpolation=None)
        plot, = ax1.plot([], [], color='red', marker='+', ms=6, linestyle='')
        self.info[ax1] = ([], plot, 'A', A)

        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(B * 0.6, interpolation=None)
        plot, = ax2.plot([], [], color='red', marker='+', ms=6, linestyle='')
        self.info[ax2] = ([], plot, 'B', B)

    def capture(self):
        while True:
            try:
                x = plt.ginput(n=1, timeout=-1)[0]
            except:
                if self.featuresOk:
                    saveInfo = {}

                    for i, k in enumerate(self.info):
                        X = self.info[k][-1]
                        coords = [(0, 0), (0, X.shape[0] - 1), (X.shape[1] - 1, 0), (X.shape[1] - 1, X.shape[0] - 1)]
                        coords.extend([c[0] for c in self.info[k][0]])
                        saveInfo[self.info[k][-2]] = coords

                    self.showTri(saveInfo)
                    plt.waitforbuttonpress()
                    plt.close()

                    if self.save:
                        self.saveInfo(saveInfo)

                    return saveInfo

                else:
                    continue

            coords = self.info[self.ax][0]
            coords.append((x, self.ax.text(0, 0, '', color='red', fontsize=7)))
            self.updateUI()

        plt.show()

    def showTri(self, info):
        d = triangulate(np.array(info['A']), np.array(info['B']), .5)

        for ax in self.info.keys():
            ax.triplot(d.points[:, 0], d.points[:, 1], d.simplices.copy())
            ax.plot(d.points[:, 0], d.points[:, 1], '+')

        self.updateUI()

    def onEnterAxis(self, event):
        self.ax = event.inaxes

    def onKeyPress(self, event):
        if event.key == 'backspace':
            coords = self.info[self.ax][0]

            if len(coords):
                c = coords.pop()
                c[1].remove()
                self.updateUI()

    def updateUI(self):
        coords, plot = self.info[self.ax][:-2]
        plot.set_data([c[0][0] for c in coords], [c[0][1] for c in coords])

        for i, c in enumerate(coords):
            c[1].set_text(str(i))
            c[1].set_position((c[0][0] + 5, c[0][1] - 5))

        coordsAll = list(self.info.values())
        coordsCountA = len(coordsAll[0][0])
        coordsCountB = len(coordsAll[1][0])
        self.featuresOk = coordsCountA == coordsCountB and coordsCountA >= 4

        self.fig.canvas.draw()

    def saveInfo(self, info):
        with open(self.save, 'wt') as file:
            json.dump(info, file, indent=2)


def computeAffine(tri1, tri2):
    xcoord1, ycoord1 = tri1[1] - tri1[0], tri1[2] - tri1[0]
    xcoord2, ycoord2 = tri2[1] - tri2[0], tri2[2] - tri2[0]
    T1 = np.linalg.inv(np.array([xcoord1, ycoord1]).T)
    T2 = np.array([xcoord2, ycoord2]).T
    return T2 @ T1


def morph(A, B, vtxA, vtxB, warpK, dissolveK):
    M = np.zeros(A.shape)

    midVtxs = warpK * vtxA + (1 - warpK) * vtxB
    midTris = scipy.spatial.Delaunay(midVtxs, qhull_options='QJ')

    interpA = interp.RectBivariateSpline(np.arange(A.shape[1]), np.arange(A.shape[0]), A.T)
    interpB = interp.RectBivariateSpline(np.arange(B.shape[1]), np.arange(B.shape[0]), B.T)

    x = np.repeat(np.arange(M.shape[1]), M.shape[0])
    y = np.tile(np.arange(M.shape[0]), M.shape[1])

    triIdx = midTris.find_simplex(np.array([x, y]).T)
    tri = midTris.simplices[triIdx]

    midTriPts = midVtxs[tri[:, 0]], midVtxs[tri[:, 1]], midVtxs[tri[:, 2]]

    bary = baryentric(np.array([x, y]).T, midTriPts[0], midTriPts[1], midTriPts[2])

    aTriPts = vtxA[tri[:, 0]], vtxA[tri[:, 1]], vtxA[tri[:, 2]]
    bTriPts = vtxB[tri[:, 0]], vtxB[tri[:, 1]], vtxB[tri[:, 2]]

    newCoordA = aTriPts[0] * bary[0] + aTriPts[1] * bary[1] + aTriPts[2] * bary[2]
    newCoordB = bTriPts[0] * bary[0] + bTriPts[1] * bary[1] + bTriPts[2] * bary[2]
    valA = interpA(newCoordA[:, 0], newCoordA[:, 1], grid=False)
    valB = interpB(newCoordB[:, 0], newCoordB[:, 1], grid=False)

    M[y, x] = dissolveK * valA + (1 - dissolveK) * valB

    return M


def morphSequence(A, B, vtxA, vtxB, framesK):
    images = []
    for i in range(1, framesK + 1):
        frac = i / framesK
        BGR = [
            morph(A[:, :, 0], B[:, :, 0], vtxA, vtxB, frac, frac).clip(0, 1),
            morph(A[:, :, 1], B[:, :, 1], vtxA, vtxB, frac, frac).clip(0, 1),
            morph(A[:, :, 2], B[:, :, 2], vtxA, vtxB, frac, frac).clip(0, 1)
        ]
        img = np.dstack(reversed(BGR))
        images.append(img)

    imgs2video(images)


def imgs2video(images):
    videoPath = 'out/seq0.mp4'
    fps = 30
    frame = images[0]
    height, width, channels = frame.shape

    fcc = cv.VideoWriter_fourcc(*'mp4v')
    vout = cv.VideoWriter(
        filename=videoPath, fourcc=fcc, fps=fps,
        frameSize=(width, height), isColor=True
    )

    for image in images:
        vout.write(np.uint8(255*image))

    vout.release()
    print('done')


def baryentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.einsum('ij, ij->i', v0, v0)
    d01 = np.einsum('ij, ij->i', v0, v1)
    d11 = np.einsum('ij, ij->i', v1, v1)
    d20 = np.einsum('ij, ij->i', v2, v0)
    d21 = np.einsum('ij, ij->i', v2, v1)

    denom = 1. / (d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) * denom
    w = (d00 * d21 - d01 * d20) * denom
    u = 1. - v - w
    return u[..., np.newaxis], v[..., np.newaxis], w[..., np.newaxis]


def doColorMorph(A, B, a, b):
    warpK = 0.5
    dissolveK = 0.5

    RGB = [
        morph(A[:, :, 0], B[:, :, 0], a, b, warpK, dissolveK).clip(0, 1),
        morph(A[:, :, 1], B[:, :, 1], a, b, warpK, dissolveK).clip(0, 1),
        morph(A[:, :, 2], B[:, :, 2], a, b, warpK, dissolveK).clip(0, 1)
    ]

    return np.dstack(RGB)


def showTriangulation(A, B, a, b):

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(A, interpolation=None)
    ax1.triplot(a[:, 0], a[:, 1], midTris.simplices)
    ax1.plot(a[:, 0], a[:, 1], '+')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(B, interpolation=None)
    ax2.triplot(b[:, 0], b[:, 1], midTris.simplices)
    ax2.plot(b[:, 0], b[:, 1], '+')

    plt.show()


def init():
    capture = False

    path1 = 'in/georgeSmall.jpeg'
    path2 = 'in/boySmall.jpeg'
    A = ops.point.normalize(plt.imread(path1))
    B = ops.point.normalize(plt.imread(path2))

    img1, img2 = os.path.basename(path1).split('.')[0], os.path.basename(path2).split('.')[0]
    coordsPath = 'out/{}-{}-v2.json'.format(img1, img2)

    if capture:
        coords = CorrespondenceUI(A, B, coordsPath).capture()
        print(coords)
    else:
        with open(coordsPath, 'rt') as file:
            coords = json.load(file)

    a, b = np.array(coords['A']), np.array(coords['B'])
    video = morphSequence(A, B, a, b, 40)

    # img = doColorMorph(A, B, a, b)
    # showTriangulation()


