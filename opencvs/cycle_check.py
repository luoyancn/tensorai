# -*- coding: utf-8 -*-

# 霍夫圆检测
# 同样需要进行极坐标的转换，最亮的一点表示圆心
# 霍夫圆检测对噪声比较敏感，需要先做中值滤波
# 1.检测边缘，发现可能的圆心
# 从候选圆心计算最佳半径大小


import cv2 as cv
import numpy as np


def detect_circles(img):
    dest = cv.pyrMeanShiftFiltering(img, 10, 100)
    gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,
                              1, 20, param1=50, param2=30)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(img, (i[0], i[1]), 2, (255, 0, 0), 2)


if __name__ == '__main__':
    src = cv.imread('data/detect_blob.png')
    detect_circles(src)
    cv.imshow('circle', src)
    cv.waitKey(0)
    cv.destroyAllWindows()