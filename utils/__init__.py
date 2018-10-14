from datetime import datetime
import cv2 as cv
import os

from .ops import *


class Logger:
    def __init__(self, path, **kwargs):
        append = kwargs.get('append', False)
        self.flush = kwargs.get('flush', True)

        if path:
            dir = os.path.dirname(path)

            if not os.path.exists(dir):
                os.makedirs(dir)

            self.file = open(path, 'a' if append else 'w')
        else:
            self.file = None

    def __call__(self, *args, **kwargs):
        if self.file:
            self.file.write(args[0] + '\n', *args[1:])
            if self.flush:
                self.file.flush()

        print(*args)

    def close(self):
        if self.file:
            self.file.close()

    def __delete__(self, instance):
        self.close()


class Profiler:
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger', print)
        self.label = kwargs.get('label', '')
        self.info = [None] * 3

    def __enter__(self):
        self.info[0] = datetime.now()
        return self.info

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.info[1] = datetime.now()
        self.info[2] = self.info[1] - self.info[0]

        if self.logger:
            label = ': {}'.format(self.label) if self.label else ''
            self.logger('[profiler{}] elapsed: {}'.format(label, self.info[2]))


def show(X, dismiss='any', **kwargs):
    title = kwargs.get('title', None)
    persist = kwargs.get('persist', False)
    key = None

    cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
    cv.imshow(title, X)
    key = cv.waitKey()

    while dismiss != 'any' and key != ord(dismiss):
        key = cv.waitKey()

    if not persist:
        cv.destroyWindow(title)

    return key

