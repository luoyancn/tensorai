# -*- coding: utf-8 -*-

# 图像的平滑处理（模糊），即使用滤波进行图像处理（卷积操作）

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


src = cv.imread('data/lena.jpg')

# 均值滤波，即简单的平均卷积操作（均值池化）
avg_blur = cv.blur(src, (7, 7))
cv.imshow('avg_blur', avg_blur)

# 方框录播，基本与均值滤波一直，但是可以选择归一化
# 第二个参数-1表示自动计算，
# normalize=True，则效果与均值滤波功能一样
# 如果normalize=False，则表示不做均值池化，只是将所有对应区域的乘加计算的
# 结果累加起来，其结果很有可能越界（超出255）。越界之后，越界的像素点全部设置为255
box_blur = cv.boxFilter(src, -1, (3, 3), normalize=True)
cv.imshow('box_blur', box_blur)

over_blur = cv.boxFilter(src, -1, (3, 3), normalize=False)
cv.imshow('box_blur_over', over_blur)

# 高斯滤波，其卷积核的数值是满足高斯分布的，即更加重视图像（区域）中间的的像素,
# 根据像素值的远近调整权重
gauss_blur = cv.GaussianBlur(src, (5, 5), 1)
cv.imshow('gauss_blur', gauss_blur)

# 中值滤波，对图像区域的像素值进行排序，选择中间值进行计算
# 中值滤波只考虑卷积核大小
center_blur = cv.medianBlur(src, 5)
cv.imshow('center_blur', center_blur)

# 将所有的水平合并在一起进行展示
res1 = np.hstack((src, avg_blur, box_blur))
res2 = np.hstack((over_blur, gauss_blur, center_blur))
# 将所有的垂直合并在一起进行展示
final = np.vstack((res1, res2))
cv.imshow('all', final)

cv.waitKey(0)
cv.destroyAllWindows()