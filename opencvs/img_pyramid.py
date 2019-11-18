# -*- coding: utf-8 -*-

# 图像金字塔
# reduce=高斯模糊+降采样, expend=或者扩大+卷积
# pyrdown:降采样；pyrup:还原

import cv2 as cv
import numpy as np


# 高斯金字塔
def gauss_pyramid(img, pyramid_level=3):
    temp = img.copy()
    pyramid_images = []
    for i in range(pyramid_level):
        dest = cv.pyrDown(temp)
        pyramid_images.append(dest)
        temp = dest.copy()
    return pyramid_images


def lapalian_pyramid(img, pyramid_level=3):
    pyramid_images = gauss_pyramid(img, pyramid_level=pyramid_level)
    lplss = []
    for i in range(pyramid_level-1, -1, -1):
        if 0 > (i - 1):
            expand = cv.pyrUp(pyramid_images[i], dstsize=img.shape[:2])
            lpls = cv.subtract(img, expand)
        else:
            expand = cv.pyrUp(pyramid_images[i],
                              dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
        lplss.append(lpls)
    return lplss


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')

    pyramid_downs = gauss_pyramid(src)
    for id, img in enumerate(pyramid_downs):
        cv.imshow('pyraid_%d' % id, img)

    lplss = lapalian_pyramid(src)
    for i, img in enumerate(lplss):
        cv.imshow('lpls_%d' %i, img)

    cv.waitKey(0)
    cv.destroyAllWindows()