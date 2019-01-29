import matplotlib.pyplot as plt
import os
import json
import numpy as np
import cv2 as cv
import scipy.spatial
from scipy import interpolate as interp
from utils import ops
from utils import Profiler


def triangulate(vtxA, vtxB, k):
    vtxM = k * vtxA + (1 - k) * vtxB
    d = scipy.spatial.Delaunay(vtxM, qhull_options='QJ')
    return d


class CorrespondenceUI:
    def __init__(self, A, B=[], save=True):
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

def saveJSON(info, folder, fileName):
    if not os.path.exists(os.path.dirname(folder)): os.mkdir(folder)
    with open(folder+fileName, 'wt') as file:
        json.dump(info, file, indent=2)

def computeAffine(tri1, tri2):
    xcoord1, ycoord1 = tri1[1] - tri1[0], tri1[2] - tri1[0]
    xcoord2, ycoord2 = tri2[1] - tri2[0], tri2[2] - tri2[0]
    T1 = np.linalg.inv(np.array([xcoord1, ycoord1]).T)
    T2 = np.array([xcoord2, ycoord2]).T
    return T2 @ T1


def makeInterp(X, k=1):
    X = X[..., np.newaxis] if len(X.shape) == 2 else X
    idx = np.arange(X.shape[-1])
    x, y = np.arange(X.shape[1]), np.arange(X.shape[0])

    interps = [
        scipy.interpolate.RectBivariateSpline(x, y, X[..., i].T, kx=k, ky=k)
        for i in idx
    ]

    def interpFunc(x, y, **kwargs):
        val = [i(x, y, **kwargs) for i in interps]
        val = np.array(val).T.squeeze()
        return val

    return interpFunc


def getMid(vtxs):
    midVtxs = np.sum(vtxs, axis=0) / vtxs.shape[0]
    midTris = scipy.spatial.Delaunay(midVtxs, qhull_options='QJ')

    return midVtxs, midTris


def morph(A, B, vtxA, vtxB, warpK, dissolveK, mid=None):
    M = np.zeros(A.shape)

    if not mid:
        midVtxs = warpK * vtxA + (1 - warpK) * vtxB
        midTris = scipy.spatial.Delaunay(midVtxs, qhull_options='QJ')
    else:
        midVtxs = mid[0]
        midTris = mid[1]

    interpA = makeInterp(A)
    interpB = makeInterp(B)

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

    M[y, x] = (dissolveK * valA + (1 - dissolveK) * valB)

    return M


def storeImgs(images, folder):
    path = 'out/{}/'.format(folder)
    if not os.path.exists(os.path.dirname(path)): os.mkdir(path)
    [cv.imwrite(path+'{}.jpg'.format(i), (im * 255).astype(np.uint8)) for i, im in enumerate(images, 0)]


def storeImg(image, folder, fileName, isColor=True):
    if not os.path.exists(os.path.dirname(folder)): os.mkdir(folder)

    if isColor :
        image = (image * 255).astype(np.uint8)
        plt.imsave(folder+'{}.jpg'.format(fileName), image)
    else:
        plt.imsave(folder+'{}.jpg'.format(fileName), image, cmap='Greys_r')


def morphSequence(A, B, vtxA, vtxB, framesK, out, makeVideo=False, isColor=True):
    images = []
    for i in range(1, framesK + 1):
        frac = i / framesK
        # img = morph(A, B, vtxA, vtxB, frac, frac).clip(0,1)
        img = warpInv2(A, vtxA, B, vtxB, 1, frac)
        if isColor : img = img[...,::-1]
        images.append(img)

        print('{}% completed'.format((i/framesK)*100))

    reverse = images.copy()
    reverse.reverse()
    images.extend(reverse)

    if makeVideo:
        storeImgs(images, out)
        imgs2video(images, out)
    else: storeImgs(images)


def imgs2video(images, out):
    videoPath = 'out/{}.mp4'.format(out)
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


def baryentricMulti(p, a, b, c):
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
    return np.array([u,v,w]).T


def doColorMorph(A, B, a, b, warpK, dissolveK):

    img = morph(A, B, a, b, warpK, dissolveK).clip(0, 1)[..., ::-1]

    return img


def showTriangulation(A, B, a, b, warpK, mid=None):

    if not mid:
        midVtxs = warpK * a + (1 - warpK) * b
        midTris = scipy.spatial.Delaunay(midVtxs, qhull_options='QJ')
    else:
        midVtxs = mid[0]
        midTris = mid[1]

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(A, interpolation=None, cmap='Greys_r')
    ax1.triplot(a[:, 0], a[:, 1], midTris.simplices)
    ax1.plot(a[:, 0], a[:, 1], '+')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(B, interpolation=None, cmap='Greys_r')
    ax2.triplot(b[:, 0], b[:, 1], midTris.simplices)
    ax2.plot(b[:, 0], b[:, 1], '+')

    plt.show()


def warpInv2(A, vtxA, B, vtxB, k, t):
    interpA = makeInterp(A, k=k)
    interpB = makeInterp(B, k=k)

    M = np.zeros_like(A)
    d = triangulate(vtxA, vtxB, t)

    x = np.repeat(np.arange(M.shape[1]), M.shape[0])
    y = np.tile(np.arange(M.shape[0]), M.shape[1])

    tidx = d.simplices
    tNum = tidx.shape[0]
    triM = d.points[tidx]
    triA = vtxA[tidx]
    triB = vtxB[tidx]

    # Compute one transform for each triangle:
    T = np.zeros((tNum, 3, 3, 2))

    basis = np.array([triM[:, 1] - triM[:, 0], triM[:, 2] - triM[:, 0]])
    basis = np.moveaxis(basis, 0, -1)
    T[:, :-1, :-1, 0] = basis
    T[:, :-1, -1, 0] = triM[:, 0]
    T[:, -1, -1, 0] = 1
    T_ = np.linalg.pinv(T[..., 0])
    T[:] = T_[..., np.newaxis]

    Ts2a = np.zeros((tNum, 3, 3))
    basis = np.array([triA[:, 1] - triA[:, 0], triA[:, 2] - triA[:, 0]])
    basis = np.moveaxis(basis, 0, -1)
    Ts2a[:, :-1, :-1] = basis
    Ts2a[:, :-1, -1] = triA[:, 0]
    Ts2a[:, -1, -1] = 1

    Ts2b = np.zeros((tNum, 3, 3))
    basis = np.array([triB[:, 1] - triB[:, 0], triB[:, 2] - triB[:, 0]])
    basis = np.moveaxis(basis, 0, -1)
    Ts2b[:, :-1, :-1] = basis
    Ts2b[:, :-1, -1] = triB[:, 0]
    Ts2b[:, -1, -1] = 1

    T[..., 0] = Ts2a @ T[..., 0]
    T[..., 1] = Ts2b @ T[..., 1]

    # Process each point:
    # tidx indexes into the right transform
    # (i.e. for the triangle the point belongs to)
    pm = np.array([x, y]).T
    tidx = d.find_simplex(pm)

    pm_ = np.column_stack((pm, np.ones(pm.shape[0])))[..., np.newaxis]
    pa = (T[tidx, ..., 0] @ pm_)
    pb = (T[tidx, ..., 1] @ pm_)

    aVal = interpA(pa[:, 0], pa[:, 1], grid=False)
    bVal = interpB(pb[:, 0], pb[:, 1], grid=False)
    M[y, x] = t * aVal + (1-t) * bVal
    return M


def autoContrast(img, i=(0, 1)):

    return i[0] + (img - img.min()) * (i[1] - i[0]) / (img.max() - img.min())


def init():
    capture = False


    path1 = 'in/family/avosAndDad.jpg'
    path2 = 'in/family/avosAndMom.jpg'
    A = ops.point.normalize(plt.imread(path1))
    B = ops.point.normalize(plt.imread(path2))
    A, B = autoContrast(A), autoContrast(B)
    warpK = 0.5
    dissolveK = 0.5
    img1, img2 = os.path.basename(path1).split('.')[0], os.path.basename(path2).split('.')[0]
    coordsPath = 'out/{}-{}.json'.format(img1, img2)

    if capture:
        coords = CorrespondenceUI(A, B, coordsPath).capture()
        print(coords)
    else:
        with open(coordsPath, 'rt') as file:
            coords = json.load(file)

    a, b = np.array(coords['A']), np.array(coords['B'])
    outPath = '{}-{}-inverse'.format(img1, img2)

    with Profiler():
        # morphSequence(A, B, a, b, 45, outPath, makeVideo=True)

        # showTriangulation(A, B, a, b, 0.5)
        # img = doColorMorph(A, B, a, b, warpK, dissolveK)
        # img = warpInv2(A, a, B, b, 1, 0.5)
        img = morph(A, B, a, b, warpK, dissolveK)
        save = 'final-v4'
        storeImg(img, 'out/family/', save, isColor=False)

    # plt.imshow(img)
    # plt.show()


