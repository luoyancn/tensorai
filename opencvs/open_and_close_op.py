# -*- coding: utf-8 -*-

# 开操作与闭操作，图像形态学的操作之一，基于膨胀与服饰操作组合而成
# 主要应用在二值图像分析当中，也可以用于灰度图像
# 开操作=腐蚀+膨胀，输入图像+结构元素，其作用是消除小的干扰区域
# 闭操作=膨胀+腐蚀，其作用是填充小的闭合区域
# 开闭操作也可以提取水平或垂直线

import cv2 as cv
import numpy as np


def open_opt(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # MORPH_RECT表示矩形元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return binary


def close_opt(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return binary


# 提取水平线
def line_opt(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # 提取垂直线
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return binary


if __name__ == '__main__':
    src = cv.imread('data/notes.png')
    cv.imshow('src', src)
    cv.imshow('open_opt', open_opt(src))
    cv.imshow('close_opt', close_opt(src))
    cv.imshow('line_opt', line_opt(src))
    # 去除干扰
    cv.imshow('example', open_opt(cv.imread('data/lines.png')))

    cv.waitKey(0)
    cv.destroyAllWindows()