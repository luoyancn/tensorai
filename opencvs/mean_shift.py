# -*- coding: utf-8 -*-

# 直方图反向投影
# 多数在HSV和RGB色彩空间
# 简单的说就是使用示例图像去查找目标

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 2D直方图
def hist2d(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 计算直方图，计算图像通道0和1的直方图(即B和G通道), 不使用mask进行遮罩
    # 最后一个参数，表示通道颜色的范围，B通道的是0-180，G通道的是0-256
    hsv_hist = cv.calcHist([img], [0,1], None, [180, 256], [0, 180, 0, 256])
    return hsv_hist


def back_projection():
    sample = cv.imread('data/opencv-logo-white.png')
    target = cv.imread('data/opencv-logo.png')
    sample_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    roiHist = cv.calcHist([sample_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    output = cv.calcBackProject(target_hsv, [0, 1], roiHist, [0, 180, 0, 256], 1)
    return sample, target, output


if __name__ == '__main__':
    #src = cv.imread('data/lena.jpg')
    #cv.imshow('src', src)

    #histres = hist2d(src)
    #cv.imshow('hist2d', histres)
    #plt.imshow(histres)
    #plt.title('2D histogram')
    #plt.show()
    sample, target, output = back_projection()
    cv.imshow('sample', sample)
    cv.imshow('target', target)
    cv.imshow('output', output)

    cv.waitKey(0)
    cv.destroyAllWindows()