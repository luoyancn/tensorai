# -*- coding: utf-8 -*-

# 模糊操作：
# 均值模糊，中值模糊，自定义模糊
# 是模糊是卷积的一种表象

import cv2 as cv
import numpy as np


def fuzzy_blur(img, kernel):
    # 均值模糊，常用于图像去噪
    dest = cv.blur(img, kernel)
    return dest


def median_blur(img, kernelsize):
    # 中值模糊，可能相当于池化？？？
    dest = cv.medianBlur(img, kernelsize)
    return dest


def custom_fuzzy(img):
    # 实际上就是卷积的操作。卷积操作可以做模糊，也可以做锐化（图像增强）
    # kernel的总和为1，做增强，总和为0， 做边缘梯度
    ad_kernel = np.array([[-1,0,1], [0, 1, 1], [-1, 0, -1]], np.float32)
    fu_kernel = np.array([[-1,0,1], [0, 1, 0], [-1, 0, 1]], np.float32)
    return cv.filter2D(img, -1, kernel=ad_kernel), \
        cv.filter2D(img, -1, kernel=fu_kernel)


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    img = src.copy()

    # 水平方向模糊，即在使用大小为15×1的卷积核对图像进行卷积操作，步长为1
    #cv.imshow('horizon_fuzzy', fuzzy_blur(img, (15, 1)))
    # 垂直方向模糊，即使用1×15的卷积核对图像进行卷积操作，步长为1
    #cv.imshow('vertical_fuzzy', fuzzy_blur(img, (1, 15)))
    # x和y方向都进行模糊，步长为1，即卷积神经网络当中常用的卷积操作
    cv.imshow('fuzze', fuzzy_blur(img, (3, 3)))
    cv.imshow('fuzzy', median_blur(img, 7))
    ad, fu = custom_fuzzy(img)
    cv.imshow('advanced', fu)
    cv.imshow('sobel', ad)

    cv.waitKey(0)
    cv.destroyAllWindows()