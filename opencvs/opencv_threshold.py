# -*- coding: utf-8 -*-

# 图像的阈值处理需要使用threshold函数，而这个函数的输入智能是单通道灰度图
# thresh表示阈值，maxval表示，当像素值超过了阈值或小于阈值，该像素点将被赋予的值
# type表示二值化操作的类型，包括了
# THRESH_BINARY: 超过阈值的部分取maxval（最大值），否则取零
# THRESH_BINARY_INV: THRESH_BINARY的反转
# THRESH_BINARY_TRUNC: 大于阈值的部分设为阈值，否则不变
# THRESH_BINARY_TOZERO: 大于阈值的部分不变，否则修改为0
# THRESH_BINARY_TO_ZERO_INV: 大于阈值的部分设置为0， 否则不变

import cv2 as cv
import matplotlib.pyplot as plt


gray_src = cv.imread('data/lena.jpg', flags=cv.IMREAD_GRAYSCALE)
_, thresh_binary = cv.threshold(gray_src, 127, 255, cv.THRESH_BINARY)
_, thresh_binary_inv = cv.threshold(gray_src, 127, 255, cv.THRESH_BINARY_INV)
_, thresh_trunc = cv.threshold(gray_src, 127, 255, cv.THRESH_TRUNC)
_, thresh_to_zero = cv.threshold(gray_src, 127, 255, cv.THRESH_TOZERO)
_, thresh_to_zero_inv = cv.threshold(gray_src, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['origin', 'thresh_binary', 'thresh_binary_inv',
         'thresh_trunc', 'thresh_to_zero', 'thresh_to_zero_inv']
images = [gray_src, thresh_binary, thresh_binary_inv,
          thresh_trunc, thresh_to_zero, thresh_to_zero_inv]

for i, k_v in enumerate(zip(titles, images)):
    plt.subplot(240+i)
    plt.imshow(k_v[1], 'gray')
    plt.title(k_v[0])
    plt.xticks([])
    plt.yticks([])

plt.show()