# -*- coding: utf-8 -*-

# 边缘提取，利用图像梯度.Canny对噪声非常敏感
# canny算法的主要步骤
# 1.图像高斯模糊
# 2.灰度转换
# 3.梯度计算
# 4.非最大信号抑制，其中，黄色区域的取值为0-22.5, 157.5-180;
# 绿色取值为22.5-67.5；蓝色为67.5-112.5；红色为112.5-157.5
# 5.高低阈值输出二值图像
# 所谓的高低阈值输出，T1，T2为阈值，凡是高于T2的都保留，低于T1的都丢弃；
# 从高于T2的像素书法，凡是大于T1而且相互连接的，都保留，最终得到一个输出的而至图像
# 高低阈值比通常为T2：T1 = 3：1 或者2：1，T2为高，T1为低


import cv2 as cv
import numpy as np


def canny_func(img):
    # 高斯模糊，降低噪声
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    # 灰度
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # 梯度计算
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    #edge_output = cv.Canny(gray, 50, 150)
    edge_output = cv.Canny(xgrad, ygrad, 40, 150)
    dest = cv.bitwise_and(img, img, mask=edge_output)

    return edge_output, dest


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    edge_output, dest = canny_func(src)
    cv.imshow('edge', edge_output)
    cv.imshow('dest', dest)

    cv.waitKey(0)
    cv.destroyAllWindows()