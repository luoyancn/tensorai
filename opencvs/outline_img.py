# -*- coding: utf-8 -*-

# 轮廓发现
# 基于图像边缘提取的基础，需要对象轮廓。边缘提取的阈值选择会影响最终轮廓的发现结果

import cv2 as cv
import numpy as np


def contours(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _contours, herjachy =  cv.findContours(
        binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, _ in enumerate(_contours):
        cv.drawContours(img, _contours, i, (0,0,255), 2)


def gray_contours(img):
    dest = cv.GaussianBlur(img, (3, 3), 0)
    gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _contours, herjachy =  cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, _ in enumerate(_contours):
        cv.drawContours(img, _contours, i, (0,0,255), 2)
        # 填充轮廓
        # cv.drawContours(img, _contours, i, (0,0,255), -1)


def binary_contours(img):
    blured = cv.GaussianBlur(img, (3, 3), 0)
    gray = cv.cvtColor(blured, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(gray, 50, 150)

    _contours, heriachy = cv.findContours(
        edge_output, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i, _ in enumerate(_contours):
        cv.drawContours(img, _contours, i, (0,0,255), -1)


if __name__ == '__main__':
    src = cv.imread('data/detect_blob.png')
    temp = src.copy()
    temp2 = src.copy()
    contours(src)
    cv.imshow('apple', src)

    gray_contours(temp)
    cv.imshow('temp', temp)
    binary_contours(temp2)
    cv.imshow('temp2', temp2)

    cv.waitKey(0)
    cv.destroyAllWindows()