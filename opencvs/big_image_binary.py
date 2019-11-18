# -*- coding: utf-8 -*-

# 大图像通常采用局部二值化，或者分块之后全局二值化的处理方式

import cv2 as cv
import numpy as np


def big_image_binary(img):
    cwight = 256
    cheight = 256
    heigh, wight = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 分块之后进行全局二值化
    for row in range(0, heigh, cheight):
        for col in range(0, wight, cwight):
            roi = gray[row:row+cheight, col:cwight+col]
            _, dest = cv.threshold(roi, 0, 255,
                                   cv.THRESH_BINARY | cv.THRESH_OTSU)
            gray[row:row+cheight, col:col+cwight] = dest
    return gray


def big_img_local_binary(img):
    cwight = 256
    cheight = 256
    heigh, wight = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 分块之后进行局部二值化
    for row in range(0, heigh, cheight):
        for col in range(0, wight, cwight):
            roi = gray[row:row+cheight, col:cwight+col]
            dest = cv.adaptiveThreshold(
                roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY, 127, 2)
            gray[row:row+cheight, col:col+cwight] = dest
    return gray


def big_image_binary_modify(img):
    cwight = 256
    cheight = 256
    heigh, wight = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 分块之后进行全局二值化，并使用空白图像过滤
    for row in range(0, heigh, cheight):
        for col in range(0, wight, cwight):
            roi = gray[row:row+cheight, col:cwight+col]
            dev = np.std(roi)
            if dev < 1:
                gray[row:row+cheight, col:col+cwight] = 255
            else:
                _, dest = cv.threshold(roi, 0, 255,
                                       cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row+cheight, col:col+cwight] = dest
    return gray


if __name__ == '__main__':
    src = cv.imread('CCCP.jpg')
    cv.imshow('cccp', src)
    cv.imshow('global_gray_cccp', big_image_binary(src))
    cv.imshow('local_gray_cccp', big_img_local_binary(src))
    cv.imshow('global_gray_modify', big_image_binary_modify(src))
    cv.waitKey(0)
    cv.destroyAllWindows()