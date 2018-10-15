import matplotlib.pyplot as plt
import os
import json
import numpy as np
import scipy.spatial
from scipy import interpolate


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
                        coords = [(0, 0), (0, X.shape[0]-1), (X.shape[1]-1, 0), (X.shape[1]-1, X.shape[0]-1)]
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
            coords.append((x, self.ax.text(0,0, '', color='red', fontsize=7)))
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

def morph(A, B, vtxA, vtB, k):
    pass

def baryentric(p, a, b, c):
    v0, v1, v2 = b-a, c-a, p-a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = 1. / (d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) * denom
    w = (d00 * d21 - d01 * d20) * denom
    u = 1. - v - w
    return u, v, w

def init():
    capture = False

    path1 = 'in/georgeSmall.jpeg'
    path2 = 'in/boySmall.jpeg'
    A = plt.imread(path1) / 255
    B = plt.imread(path2) / 255
    img1, img2 = os.path.basename(path1).split('.')[0], os.path.basename(path2).split('.')[0]
    coordsPath = 'out/{}-{}-v2.json'.format(img1, img2)

    if capture:
        coords = CorrespondenceUI(A, B, coordsPath).capture()
        print(coords)
    else:
        with open(coordsPath, 'rt') as file:
            coords = json.load(file)

    a, b = np.array(coords['A']), np.array(coords['B'])
    t = 0.5
    midVtxs = t * a + (1 - t) * b
    midTris = scipy.spatial.Delaunay(midVtxs, qhull_options='QJ')
    interpA = interpolate.interp2d(np.arange(A.shape[1]), np.arange(A.shape[0]), A[:,:,0], kind='linear')
    interpB = interpolate.interp2d(np.arange(B.shape[1]), np.arange(B.shape[0]), B[:,:,0], kind='linear')

    M = np.zeros(A.shape)

    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            triIdx = midTris.find_simplex(np.array([x, y]))
            tri = midTris.simplices[triIdx]
            midTriPts = midVtxs[tri[0]], midVtxs[tri[1]], midVtxs[tri[2]]
            bary = baryentric((x, y), midTriPts[0], midTriPts[1], midTriPts[2])

            aTriPts = (a[tri[0]], a[tri[1]], a[tri[2]])
            bTriPts = (b[tri[0]], b[tri[1]], b[tri[2]])

            newCoordA = aTriPts[0] * bary[0] + aTriPts[1] * bary[1] + aTriPts[2] * bary[2]
            newCoordB = bTriPts[0] * bary[0] + bTriPts[1] * bary[1] + bTriPts[2] * bary[2]
            valA = interpA(*newCoordA)
            valB = interpB(*newCoordB)

            M[y, x] = t*valA + (1-t)*valB

    plt.imshow(M)
    plt.show()


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
