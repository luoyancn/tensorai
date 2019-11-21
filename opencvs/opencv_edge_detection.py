# -*- coding: utf-8 -*-

# 图像的边缘检测
# Canny边缘检测：
# 1.使用高斯录播其，进行图像平滑/模糊处理，并且进行噪声过滤
# 2.计算每个像素点的梯度强度和方向
# 3.应用非极大值一直，消除边缘检测带来的杂散响应
# 4.应用双阈值检测确定真实的和潜在的边缘
# 5.通过抑制弱边缘，最终完成边缘检测

import cv2 as cv
import numpy as np


src = cv.imread('data/lena.jpg', flags=cv.IMREAD_GRAYSCALE)
v1 = cv.Canny(src, 80, 150)
v2 = cv.Canny(src, 50, 100)
res = np.hstack((v1, v2))
cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()