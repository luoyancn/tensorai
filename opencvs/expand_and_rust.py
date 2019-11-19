# -*- coding: utf-8 -*-

# 图像形态学：膨胀与腐蚀
# 灰度图像与而至图像处理的重要手段

# 膨胀（Dilate）：3×3的结构元素/模板，支持任意形状的结构元素，可以认为是最大值滤波
# 腐蚀（Erode）：3×3的结构元素/模板，支持任意形状的结构元素

import cv2 as cv
import numpy as np


def erode_and_dilate_func(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # 腐蚀，即形态缩小
    _erode = cv.erode(binary, kernel)
    # 膨胀，即形态放大
    _dilate = cv.dilate(binary, kernel)
    return _erode, _dilate


if __name__ == '__main__':
    src = cv.imread('data/numbers.png')
    _erodo, _dilate = erode_and_dilate_func(src)
    cv.imshow('rust_numbers', _erodo)
    cv.imshow('expand_numbers', _dilate)

    # 膨胀和腐蚀也可以用在rgb图像上
    rgb_src = cv.imread('data/ml.png')
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cv.imshow('rgb_src', rgb_src)
    cv.imshow('rgb_rust', cv.erode(rgb_src, kernel))
    cv.imshow('rgb_expand', cv.dilate(rgb_src, kernel))

    cv.waitKey(0)
    cv.destroyAllWindows()