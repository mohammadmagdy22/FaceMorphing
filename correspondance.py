import matplotlib.pyplot as plt
import os
import json
import numpy as np
import scipy.spatial


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
        ax1.imshow(A, interpolation=None)
        plot, = ax1.plot([], [], color='red', marker='+', ms=6, linestyle='')
        self.info[ax1] = ([], plot, 'A')

        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(B, interpolation=None)
        plot, = ax2.plot([], [], color='red', marker='+', ms=6, linestyle='')
        self.info[ax2] = ([], plot, 'B')


    def capture(self):
        while True:
            try:
                x = plt.ginput(n=1, timeout=-1)[0]
            except:
                if self.featuresOk:
                    plt.close()

                    saveInfo = {}

                    for i, k in enumerate(self.info):
                        saveInfo[self.info[k][-1]] = [c[0] for c in self.info[k][0]]

                    if self.save:
                        self.saveInfo(saveInfo)

                    return saveInfo
                else:
                    continue

            coords = self.info[self.ax][0]
            coords.append((x, self.ax.text(0, 0, '', color='red', fontsize=7)))
            self.updateUI()

        plt.show()

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
        coords, plot = self.info[self.ax][:-1]
        plot.set_data([c[0][0] for c in coords], [c[0][1] for c in coords])

        for i, c in enumerate(coords):
            c[1].set_text(str(i))
            c[1].set_position((c[0][0] + 5, c[0][1] - 5))

        coordsAll = list(self.info.values())
        coordsCountA = len(coordsAll[0][0])
        coordsCountB = len(coordsAll[1][0])
        self.featuresOk = coordsCountA == coordsCountB and coordsCountA >= 2

        self.fig.canvas.draw()

    def saveInfo(self, info):
        with open(self.save, 'wt') as file:
            json.dump(info, file, indent=2)


def init():
    capture = False

    path1 = 'in/george.jpg'
    path2 = 'in/boy.jpg'
    A = plt.imread(path1)
    B = plt.imread(path2)
    img1, img2 = os.path.basename(path1).split('.')[0], os.path.basename(path2).split('.')[0]
    coordsPath = 'out/{}-{}.json'.format(img1, img2)

    if capture:
        coords = CorrespondenceUI(A, B, coordsPath).capture()
        print(coords)
    else:
        with open(coordsPath, 'rt') as file:
            coords = json.load(file)

    # compute mean shape
    a, b = np.array(coords['A']), np.array(coords['B'])
    k = 0.5
    c = k * a + (1 - k) * b
    d = scipy.spatial.Delaunay(c, qhull_options='QJ')

    # plot both images
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(A, interpolation=None)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(B, interpolation=None)

    # plot mean shape on top of images
    ax1.triplot(c[:, 0], c[:, 1], d.simplices.copy())
    ax1.plot(c[:, 0], c[:, 1], '+')
    plt.gca().invert_yaxis()
    ax2.triplot(c[:, 0], c[:, 1], d.simplices.copy())
    ax2.plot(c[:, 0], c[:, 1], '+')
    plt.gca().invert_yaxis()

    plt.show()

