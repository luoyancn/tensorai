# -*- coding: utf-8 -*-

# 边缘保留滤波
# 可以使用高斯双边和均值迁移，通常是美颜的常用方式

import cv2 as cv


# 高斯双边模糊，保留边缘。但是高斯模糊不会
def guass_bi(img):
    return cv.bilateralFilter(img, 0, 100, 15)


# 均值迁移, 类似油画效果
def gauss_bi(img):
    return cv.pyrMeanShiftFiltering(img, 10, 50)


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    cv.imshow('src', src)
    cv.imshow('gauss_bi', guass_bi(src))
    cv.imshow('guass_bi', gauss_bi(src))

    cv.waitKey(0)
    cv.destroyAllWindows()