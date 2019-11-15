# -*- coding: utf-8 -*-

# 高斯模糊：主要依据是高斯分布


import cv2 as cv
import numpy as np


def gaussian_noise(img):


    def _clamp(pv):
        if pv > 255:
            return 255
        elif pv <0:
            return 0
        else:
            return pv


    height, wight, channels = img.shape
    for row in range(height):
        for col in range(wight):
            # 生成3个标准差/方差在20的随机数
            s = np.random.normal(0, 20, 3)
            # 分离每个通道的每个像素点
            blue = img[row, col, 0]
            green = img[row, col, 1]
            red = img[row, col, 2]
            # 添加随机噪声
            img[row, col, 0] = _clamp(blue + s[0])
            img[row, col, 1] = _clamp(green + s[1])
            img[row, col, 2] = _clamp(red + s[2])
    return img


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    img = src.copy()
    cv.imshow('img', img)
    cv.imshow('gauss', gaussian_noise(img))

    # OpenCV自带的高斯模糊，即毛玻璃效果
    dest = cv.GaussianBlur(img, (0, 0), 10)
    cv.imshow('Gauss_Blur', dest)

    cv.waitKey(0)
    cv.destroyAllWindows()