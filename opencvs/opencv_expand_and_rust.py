# -*- coding: utf-8 -*-

# 形态学，膨胀与腐蚀
# 处理二值图像/灰度图像的噪音和毛刺

import cv2 as cv
import numpy as np


src = cv.imread('data/LinuxLogo.jpg')

kernel = np.ones((3, 3), np.uint8)

# 膨胀操作
expand = cv.dilate(src, kernel, iterations=1)
#cv.imshow('expand', expand)

# 腐蚀操作
erosion = cv.erode(src, kernel, iterations=2)
#cv.imshow('erosion', erosion)

# 开运算与闭运算
# 开： 先腐蚀，再膨胀
opening = cv.morphologyEx(src, cv.MORPH_OPEN, kernel)
#cv.imshow('opening', opening)

# 闭：先膨胀，再腐蚀
closing = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel)
#cv.imshow('closing', closing)

res1 = np.hstack((expand, erosion))
res2 = np.hstack((opening, closing))
final = np.vstack((res1, res2))
cv.imshow('final_all', final)

# 梯度运算，实际上就是膨胀-腐蚀
# 可以获取形态的边缘与轮廓
gradient = cv.morphologyEx(src, cv.MORPH_GRADIENT, kernel)
cv.imshow('gradient', gradient)

# 顶帽（礼帽）：原始输入-开运算结果
tophat = cv.morphologyEx(src, cv.MORPH_TOPHAT, kernel)
cv.imshow('tophat', tophat)

# 黑帽：闭运算-原始输入
blackhat = cv.morphologyEx(src, cv.MORPH_BLACKHAT, kernel)
cv.imshow('blackhat', blackhat)

cv.waitKey(0)
cv.destroyAllWindows()