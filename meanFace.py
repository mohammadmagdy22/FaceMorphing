import matplotlib.pyplot as plt
from os import listdir
from utils import point as p
from utils import Profiler
import numpy as np
import morph
from morph import makeInterp
from utils import show
import scipy


def getPoints(path):
    pts = []
    with open(path, 'r') as fp:
        for cnt, line in enumerate(fp):
            if cnt < 3 or cnt > 48: continue
            pts.append([float(s) for s in line.split()])

    return np.array(pts)

def warpBaryMulti(imgs, vtxs, interpk, warpk):
    numOfImgs = imgs.shape[0]
    interps = [makeInterp(imgs[i], interpk) for i in range(numOfImgs)]
    M = np.zeros_like(imgs[0])

    vtxM = np.mean(vtxs, axis=0)
    delunay = scipy.spatial.Delaunay(vtxM, qhull_options='QJ')

    x = np.repeat(np.arange(M.shape[1]), M.shape[0])
    y = np.tile(np.arange(M.shape[0]), M.shape[1])

    triIdx = delunay.simplices
    numOfTris = triIdx.shape[0]
    triM = np.vstack((delunay.points[triIdx] for _ in range(numOfImgs)))
    T = np.vstack((vtxs[i, triIdx] for i in range(numOfImgs)))
    T = np.swapaxes(T, 1, 2)

    vtxM = np.array([x, y]).T
    triIdx = delunay.find_simplex(vtxM)
    vtxM = np.vstack([vtxM] * numOfImgs)
    triIdx = np.hstack([triIdx + numOfTris * i for i in range(numOfImgs)])

    pm_b = morph.baryentricMulti(vtxM, triM[triIdx, 0], triM[triIdx, 1], triM[triIdx, 2])[..., np.newaxis]
    pi = T[triIdx] @ pm_b
    n = x.size

    baryImgs = np.array([
        interps[i](pi[n * i: n * i + n][:, 0], pi[n * i: n * i + n][:, 1], grid=False)
        for i in range(numOfImgs)
    ])

    M[y, x] = np.mean(baryImgs, axis=0)
    return M, delunay.points.tolist()

def daisyChainMulti(images, points):
    midsvtx, midstri = morph.getMid(points)
    A = images[0]
    vtxA = points[0]
    for i in range(1, len(images)):
        B, vtxB = images[i, :, :], points[i, :, :]

        warpK = (1 / (i + 1))
        dissolveK = warpK
        # morph.showTriangulation(A, B, vtxA, vtxB, warpK, (midsvtx, midstri))
        # plt.imshow(A, cmap='Greys_r')

        A = morph.morph(A, B, vtxA, vtxB, warpK, dissolveK, (midsvtx, midstri))
        vtxA = warpK * vtxA + (1 - warpK) * vtxB

        # if i%5==0:
        #     print('{}% completed'.format((i/len(images))*100))
        # morph.showTriangulation(A, B, vtxA, vtxB, warpK, (midsvtx, midstri))

    plt.imshow(A, cmap='Greys_r')
    plt.show()

def init():

    with Profiler():
        pathFaces = 'in/brazil_subset/'
        pathPoints = 'in/brazil_faces_pts/'
        outPath = 'out/brazil_faces/'
        fileName = 'meanFace'
        images = []
        points = []
        for file in listdir(pathFaces):
            if 'a' in file:
                img = p.normalize(plt.imread(pathFaces + file))
                pts = getPoints(pathPoints + file.split('.')[0] + '.pts')
                corners = [(0, 0), (0, img.shape[0] - 1), (img.shape[1] - 1, 0), (img.shape[1] - 1, img.shape[0] - 1)]
                pts = np.append(corners, pts, axis=0)
                points.append(pts)
                images.append(img)

        images = np.stack(images)
        points = np.stack(points)

        # daisyChainMulti(images, points)
        A, vtxA = warpBaryMulti(images, points, 1, 1)

        morph.saveJSON(vtxA, outPath, fileName)
        morph.storeImg(A, outPath, fileName)

        plt.imshow(A, cmap='Greys_r')
        plt.show()

