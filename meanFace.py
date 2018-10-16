import matplotlib.pyplot as plt
from os import listdir
from utils import point as p
import numpy as np


def getPoints(path):
    pts = []
    with open(path, 'r') as fp:
        for cnt, line in enumerate(fp):
            if cnt < 3 or cnt > 48: continue
            pts.append([float(s) for s in line.split()])

    return np.array(pts)

def init():

    pathFaces = 'in/brazil_faces/'
    pathPoints = 'in/brazil_faces_pts/'
    images = np.array([])
    points = np.array([])

    for file in listdir(pathFaces):

        if 'a' in file:
            img = p.normalize(plt.imread(pathFaces + file))
            pts = getPoints(pathPoints + file.split('.')[0] + '.pts')


