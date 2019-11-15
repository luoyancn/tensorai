# -*- coding: utf-8 -*-

# 直方图均衡化
# OpenCV当中，直方图均衡化都是针对灰度图像
# 调整对比度，图像增强的手段


import cv2 as cv
import numpy as np


# 全局的直方图均质化
def mean_hist(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.equalizeHist(gray)


# 局部的直方图均值化
def clahe_hist(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)


if __name__ == '__main__':
    src = cv.imread('data/pca_test1.jpg')
    cv.imshow('src', src)

    cv.imshow('mean_hist', mean_hist(src))
    cv.imshow('clahe_hist', clahe_hist(src))

    cv.waitKey(0)
    cv.destroyAllWindows()