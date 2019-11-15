# -*- coding: utf-8 -*-

# 图像的二值化：将rgb图像转换为灰度图像，再将每个像素点的值约束在0-1之间, 0为白色，1为黑色
# 二值化的方法有2种：全局阈值和局部阈值

import cv2 as cv
import numpy as np


# 全局阈值
def threshold(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.THRESH_BINARY为必须的，后续的cv.THRESH_OTSU表示二值化的方法
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 全局阈值化。当图像的直方图只有一个波峰的时候，效果比较好，对于医学图像效果比较好，其他场景不太适用
    ret_, global_binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    # 自定义阈值进行阈值化，指定min或者max的值，最后的参数只是用THRESH_BINARY
    _ret, custom_binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # 颜色颠倒反向
    __ret, reverse_binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # 截断，大于128的，直接变为128；反之，不变？
    _ret_, trunc_binary = cv.threshold(gray, 128, 255, cv.THRESH_TRUNC)

    # 截断，小于128的，直接变为0；
    _ret_, zero_binary = cv.threshold(gray, 128, 255, cv.THRESH_TOZERO)

    return binary, global_binary, custom_binary, reverse_binary, trunc_binary, zero_binary


# 局部阈值，实际上就是自适应二值化
def local_threshold(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯阈值，即图像中重要部分的权重占比更大的阈值化, 倒数第二个参数为blocksize，该参数必须为奇数
    gauss_c = cv.adaptiveThreshold(gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    # 均值阈值，对所有像素求均值，每一个像素的值如果大于均值则为1，小于均值则为0
    # 10表示一个水位线，均值和像素值的差异 > 10， 则像素值最终为1， 反之为0
    mean_c = cv.adaptiveThreshold(gray, 255,
        cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    return gauss_c, mean_c


def custom_threshold(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    reshape_gray = np.reshape(gray, [1, height*width])
    mean = reshape_gray.sum() / (height*width)
    _ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    return binary


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    binary, global_binary, custom_binary, reverse_binary, trunc_binary, zero_binary = threshold(src)
    cv.imshow('binary', binary)
    cv.imshow('global', global_binary)
    cv.imshow('custom', custom_binary)
    cv.imshow('reverse', reverse_binary)
    cv.imshow('trunc', trunc_binary)
    cv.imshow('zero', zero_binary)

    gauss_c, mean_c = local_threshold(src)
    cv.imshow('gauss', gauss_c)
    cv.imshow('mean', mean_c)

    cv.imshow('custom_binary', custom_threshold(src))
    cv.waitKey(0)
    cv.destroyAllWindows()