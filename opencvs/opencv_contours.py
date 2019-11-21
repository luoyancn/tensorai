# -*- coding: utf-8 -*-

# 图像轮廓，需要使用findContours，其中mode表示轮廓的检索模式，method表示轮廓的逼近方法
# mode的取值
# RETR_EXTERNAL: 只检测最外面的轮廓
# RETR_LIST: 检索所有的轮廓，并保存到链表当中
# RETR_CCOMP: 检索所有轮廓，并组织为2层：顶层为外部边界，第二层为空洞边界
# RETR_TREE: 检索所有轮廓，并重构嵌套轮廓的层次

# method的取值
# CHAIN_APPROX_NONE: 以Freeman链码的方式输出轮廓，所有其他方法输出多边形
# CHAIN_APPROX_SIMPLE: 压缩水平、垂直和斜的部分，就只保留终点部分

# 通常情况下，为了提高准确率，图像轮廓的检测与提取，都是使用二值图像进行

import cv2 as cv
import numpy as np


# 可以直接使用读取灰度图像的方式，简化代码
src = cv.imread('data/pic1.png')
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#gray = cv.imread('data/lena.jpg', flags=cv.IMREAD_GRAYSCALE)
# _, gray_thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# cv.imshow('gray_thresh', gray_thresh)

_, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# 获取轮廓信息、层次信息
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

draw_img = src.copy()
# 第3个参数-1表示所有的轮廓，最后一个参数2表示线条的宽度
res = cv.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
cv.imshow('res', res)

# 轮廓特征，计算轮廓面积,周长
area, arc_lenth = cv.contourArea(
    contours[0]), cv.arcLength(contours[0], closed=True)
print(area, arc_lenth)

cnt = contours[6]
# 轮廓的外接矩形,获得轮廓外接矩形的起始坐标和长宽
x_start, y_start, wight, height = cv.boundingRect(cnt)
# 轮廓的外接圆形，获得轮廓的外接圆的圆心坐标与半径
(x_center, y_center), radius = cv.minEnclosingCircle(cnt)
# 绘制矩形
rect_img = cv.rectangle(src, (x_start, y_start),
                        (x_start+wight, y_start+height), (2, 255, 0), 2)
# 绘制圆形
res_img = cv.circle(rect_img, (int(x_center), int(y_center)),
                    int(radius), (0, 0, 255), 2)
cv.imshow('rect_outer', res_img)

# 轮廓近似
img = cv.imread('data/pic1.png')
gray_ = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh_ = cv.threshold(gray_, 127, 255, cv.THRESH_BINARY)
contours_, hierarchy_ = cv.findContours(
    thresh_, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnt = contours_[6]
epsilon = 0.02 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)
res__ = cv.drawContours(img.copy(), [approx], -1, (0, 0, 255), 2)
cv.imshow('__res', res__)

cv.waitKey(0)
cv.destroyAllWindows()