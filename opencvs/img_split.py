# -*- coding: utf-8 -*-

# 分水岭算法：通常用于图像分割
# 基于距离变幻，查找边缘
# 具体的流程是，输入图像->灰度处理->二值化->距离变换
# -> 寻找种子->程程marker->分水岭变换->输出图像

# 需要进行去噪，形态变换


import cv2 as cv
import numpy as np


def watershed(img):
    blurred = cv.pyrMeanShiftFiltering(img, 10, 100)

    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=2)

    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)

    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)

    surface_bg = np.uint8(surface)
    unkown = cv.subtract(sure_bg, surface_bg)
    ret, markers = cv.connectedComponents(surface)

    markers = markers + 1
    markers[unkown==255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers==-1] = [0, 0, 255]


if __name__ == '__main__':
    src = cv.imread('data/detect_blob.png')
    watershed(src)
    cv.imshow('waterd', src)


    cv.waitKey(0)
    cv.destroyAllWindows()