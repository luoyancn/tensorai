# -*- coding: utf-8 -*-

# 图像直方图

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def image_hist(img):
    colors = ('blue', 'green', 'red')
    # 计算图像的直方图
    for i, color in enumerate(colors):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    cv.imshow('src', src)
    plot(src)
    image_hist(src)

    cv.waitKey(0)
    cv.destroyAllWindows()