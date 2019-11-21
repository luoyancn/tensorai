# -*- coding: utf-8 -*-

# 模板匹配
# 其原理与卷积类似。模板在原图从原点进行滑动，计算模板与被覆盖区域的差别，然后将每次计算的
# 结果放入一个结果矩阵，作为输出。原图为A*B，模板为a*b，则输出为(A-a+1)*(B-b+1)

# TM_SQDIFF: 计算平方差异，差值越小，相关性越大
# TM_CCORR: 计算相关性，值越大，相关性越大
# TM_CCOEFF: 计算相关系数，系数越大，相关性越大
# TMSQDIFF_NORMED: 计算归一化平方差异，值越接近0， 相关性越大
# TM_CCORR_NORMED: 计算归一化相关性，值越接近1， 相关性越大
# TM_CCOEFF_NORMED：计算归一化相关系数，值越接近1， 相关性越大

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


src = cv.imread('data/pic1.png')
tpl = cv.imread('data/templ.png')
tpl_wight, tpl_height = tpl.shape[:2]

methods = [
    cv.TM_SQDIFF, cv.TM_CCORR, cv.TM_CCOEFF,
    cv.TM_SQDIFF_NORMED ,cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED
]

for meth in methods:
    img = src.copy()
    res = cv.matchTemplate(img, tpl, meth)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if meth in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + tpl_wight, top_left[1] + tpl_height)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121)
    plt.imshow(res, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# 多目标匹配，也可以直接使用RGB图像进行对比
img_rgb = cv.imread('data/football.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
tpl = cv.imread('data/tpl.png')
tpl = cv.cvtColor(tpl, cv.COLOR_BGR2GRAY)
height, wight = tpl.shape[:2]
_res = cv.matchTemplate(img_gray, tpl, cv.TM_CCOEFF_NORMED)
threshold = 0.99
loc = np.where(_res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0]+wight, pt[1]+height)
    cv.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv.imshow('img_rgb', img_rgb)

cv.waitKey(0)
cv.destroyAllWindows()