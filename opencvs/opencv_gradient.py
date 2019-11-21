# -*- coding: utf-8 -*-

# 梯度计算
# 通常是边缘计算或者检测

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


src = cv.imread('data/lena.jpg', flags=cv.IMREAD_GRAYSCALE)

# 梯度计算是在x/y方向, ddepth通常为-1， 表示输入和输出的一样的
xdest = cv.Sobel(src, cv.CV_64F, 1, 0, ksize=3)
# 转换为正数
xdest = cv.convertScaleAbs(xdest)
ydest = cv.Sobel(src, cv.CV_64F, 0, 1, ksize=3)
ydest = cv.convertScaleAbs(ydest)
# 可以同时进行x和y方向的梯度计算，但是不建议，而是使用分别计算再求和的方式
bad_xydest = cv.Sobel(src, cv.CV_64F, 1, 1, ksize=3)
bad_xydest = cv.convertScaleAbs(bad_xydest)

# 合并x/y的梯度计算
good_xydest = cv.addWeighted(xdest, 0.5, ydest, 0.5, 0)

titles = ['xdest', 'ydest', 'bad_xydest', 'good_xydest']
images = [xdest, ydest, bad_xydest, good_xydest]
for i, kv in enumerate(zip(titles, images)):
    plt.subplot(2, 2, 1+i)
    plt.imshow(kv[1], 'gray')
    plt.title(kv[0])
    plt.xticks([])
    plt.yticks([])
plt.show()

# scharr算子
scharrrx = cv.Scharr(src, cv.CV_64F, 1, 0)
scharrrx = cv.convertScaleAbs(scharrrx)
scharrry = cv.Scharr(src, cv.CV_64F, 0, 1)
scharrry = cv.convertScaleAbs(scharrry)
scharrrxy = cv.addWeighted(scharrrx, 0.5, scharrry, 0.5, 0)

# 拉普拉斯算子的梯度计算
laplacian = cv.Laplacian(src, cv.CV_64F)
laplacian = cv.convertScaleAbs(laplacian)
res = np.hstack((scharrrxy, laplacian, good_xydest))
cv.imshow('vs', res)
cv.waitKey(0)
cv.destroyAllWindows()