# -*- coding: utf-8 -*-

# 直线检测
# 通常使用霍夫直线变幻；其前提是边缘检测已经完成
# 需要进行平面空间到极坐标空间的转换

import cv2 as cv
import numpy as np


def line_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 40, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        # 极坐标转换
        rho, theta = line[0]
        _a = np.cos(theta)
        _b = np.sin(theta)
        x0 = _a * rho
        y0 = _b * rho
        x1 = int(x0 + 1000*(-_b))
        y1 = int(y0 + 1000*(_a))
        x2 = int(x0 - 1000*(-_b))
        y2 = int(y0 - 1000*(_a))
        # 划线
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def line_detect_possible(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 40, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/100, 100,
                           minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


if __name__ == '__main__':
    src = cv.imread('data/building.jpg')

    #line_detection(src)
    line_detect_possible(src)
    cv.imshow('lined', src)

    cv.waitKey(0)
    cv.destroyAllWindows()