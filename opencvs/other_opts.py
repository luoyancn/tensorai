# -*- coding: utf-8 -*-

# 其他形态学的操作，包括顶帽，黑帽以及形态学梯度
# 顶帽是原图像与开操作之间的差值图像
# 黑帽是闭操作与原图像之间的差值图像

# 基本梯度：使用膨胀之后的图像减去腐蚀后的图像得到的差值图像
# 内部梯度：原图监狱腐蚀之后的图像的差值图像
# 外部梯度：图像膨胀之后减去原图的差值图像

import cv2 as cv
import numpy as np


def top_hat(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,  (15, 15))
    dest = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    # 增加亮度
    cimg = np.array(gray.shape, np.uint8)
    cimg = 100
    dest = np.add(dest, cimg)
    return dest


def black_hat(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,  (15, 15))
    dest = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    # 增加亮度
    cimg = np.array(gray.shape, np.uint8)
    cimg = 100
    dest = np.add(dest, cimg)
    return dest


def binary_hat_opts(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,  (15, 15))
    dest = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    return dest


def img_grad(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,  (3, 3))
    dest = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
    return dest


def img_grad_internal(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,  (3, 3))
    dm = cv.dilate(img, kernel)
    em = cv.erode(img, kernel)
    dest1 = cv.subtract(img, em)
    dest2 = cv.subtract(dm, img)
    return dest1, dest2


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    cv.imshow('top_hat', top_hat(src))
    cv.imshow('black_hat', black_hat(src))
    cv.imshow('black_hat_binary', binary_hat_opts(src))
    cv.imshow('grad', img_grad(src))

    img1, img2 = img_grad_internal(src)
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)

    cv.waitKey(0)
    cv.destroyAllWindows()