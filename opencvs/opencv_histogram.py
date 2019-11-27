# -*- coding: utf-8 -*-

# 图像的直方图

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

src = cv.imread('data/lena.jpg')
# 原始图像，为uint8或者float32类型
# 第二个参数表示channel，即图像的通道。如果是灰度图，则是0
# 如果是rgb图像，则可以是[0], [1] [2]
# mask表示掩模图像。通常的，统计全部的直方图，则设置为None；统计一部分则使用掩模图像
# histSize： BIN的数量
# ranges：像素的范围
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
hist = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.hist(gray.ravel(), 256)

# 统计每个通道的直方图结果
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    histr = cv.calcHist([src], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])

# 掩模图像
mask = np.zeros(src.shape[:2], np.uint8)
mask[100:300, 100:400] = 255 # 需要保留的部分，设置为255，即白色
cv.imshow('mask_img', mask)
masked_img = cv.bitwise_and(src, src, mask=mask)
cv.imshow('masked_img', masked_img)

for i, col in enumerate(('c','y', 'm')):
    histr = cv.calcHist([src], [i], mask, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])

plt.show()

# 原始的灰度图的直方图
plt.hist(gray.ravel(), 256)
plt.show()
# 直方图均衡化
equ = cv.equalizeHist(gray)
plt.hist(equ.ravel(), 256)
plt.show()
cv.imshow('gray_vs_equ', np.hstack((gray, equ)))

# 自适应直方图均衡化
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(gray)
cv.imshow('gray_vs_equ', np.hstack((gray, equ, res_clahe)))

cv.waitKey(0)
cv.destroyAllWindows()