# -*- coding: utf-8 -*-

# RGB的直方图以及直方图比较

import cv2 as cv
import numpy as np


def create_rgb_hist(img):
    height, wight, channel = img.shape
    rgb_bist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256/8
    for row in range(height):
        for col in range(wight):
            blue = img[row, col, 0]
            green = img[row, col, 1]
            red = img[row, col, 2]
            index = np.int(blue/bsize)*16*16 + np.int((green/bsize) * 16) + np.int(red/bsize)
            rgb_bist[np.int(index), 0] = rgb_bist[np.int(index), 0] + 1
    return rgb_bist


def hist_compare(img1, img2):
    hist1 = create_rgb_hist(img1)
    hist2 = create_rgb_hist(img2)
    # 巴氏距离越大，说明图像的相似度越小
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    # 相关性越大，说明图像的相似度越大
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    # 卡方越大，说明图像的相似度越小，并不太常用
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print('%r\t%r\t%r' %(match1, match2, match3))


if __name__ == '__main__':
    img1 = cv.imread('data/lena.jpg')
    img2 = cv.imread('data/lena_tmpl.jpg')
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)

    hist_compare(img1, img2)

    cv.waitKey(0)
    cv.destroyAllWindows()