# -*- coding: utf-8 -*-

# 图像的梯度，即2个像素之间的差值，而这些差值组成的图，即是梯度图
# 像素差距的导数最大的地方，就是图像的边缘
# 一阶导数与soble算子，通常用于求取图像边缘处理
# 二阶导数与拉普拉斯算子，二阶导数，就是对像素差距的导数再求导数
# 结果的最小值附近就是图像的边缘
# 拉布拉斯算子与soble算子，其所有项的和为0


import cv2 as cv
import numpy as np


def soble_func(img):
    # x 方向的梯度，需要注意是第二个参数，
    # cv_8u就是256的，进行计算之后，取值会超出256，结果导致溢出
    grad_x = cv.Sobel(img, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(img, cv.CV_32F, 0, 1)

    # 将计算的结果全部转换为大于0的数值，负数转换为其绝对值
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradx, grady, gradxy


# 索贝算子的增强版，能够得到更加强烈的图像边缘
def advanced_soble_scharr(img):
    grad_x = cv.Scharr(img, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)

    # 将计算的结果全部转换为大于0的数值，负数转换为其绝对值
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradx, grady, gradxy


def lapalian_func(img):
    #dest = cv.Laplacian(img, cv.CV_32F)
    #return cv.convertScaleAbs(dest)
    # 下列代码是拉普拉斯算子的实质
    # 自定义的算子，可以在下方进行修改
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dest = cv.filter2D(img, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dest)
    return lpls


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    gradx, grady, gradxy = soble_func(src)
    cv.imshow('gradx', gradx)
    cv.imshow('grady', grady)
    cv.imshow('gradxy', gradxy)

    scharrx, scharry, scharrxy = advanced_soble_scharr(src)
    cv.imshow('scharrx', scharrx)
    cv.imshow('scharry', scharry)
    cv.imshow('scharrxy', scharrxy)

    cv.imshow('laplas', lapalian_func(src))

    cv.waitKey(0)
    cv.destroyAllWindows()