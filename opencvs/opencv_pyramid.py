# -*- coding: utf-8 -*-

# 图像金字塔：高斯金字塔，拉普拉斯金字塔
# 高斯金字塔：
# 向下采样方法（缩小）:与高斯内核卷积，将所有偶数行和列去除
# 向上采样方法（放大）:将图像在每个方向扩大为原来的2被，新增的行和列以0填充
# 使用先前同样的内核（乘以4）与放大后的图像进行卷积，获得近似值

import cv2 as cv
import numpy as np

src = cv.imread('data/lena.jpg')
up_pyramid = cv.pyrUp(src)
cv.imshow('up', up_pyramid)

down_pyraid = cv.pyrDown(src)
cv.imshow('down', down_pyraid)

# 拉普拉斯金字塔
# target = input - pyrup(pyrdown(input))
# 低通滤波->缩小尺寸->放大尺寸->图像相减

down_up = cv.pyrUp(down_pyraid)
laplas_pyramid = src - down_up
cv.imshow('laplas', laplas_pyramid)

cv.waitKey(0)
cv.destroyAllWindows()