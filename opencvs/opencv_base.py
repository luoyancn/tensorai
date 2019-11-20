# -*- coding: utf-8 -*-

# opencv读取图像是以BGR格式进行的，并不是RGB

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_img(img, name, showall=True):
    cv.imshow(name, img)
    if not showall:
        cv.waitKey(0)
        cv.destroyWindow(name)


if __name__ == '__main__':
    rgb_img = cv.imread('data/lena.jpg')
    # 灰度图像
    gray_img = cv.imread('data/lena.jpg', flags=cv.IMREAD_GRAYSCALE)
    show_img(rgb_img, 'rgb')
    print('rgb image shape ' , (rgb_img.shape))
    show_img(gray_img, 'gray')
    print('gray image shape ' , (gray_img.shape))
    #cat = rgb_img[:228, :228,:]
    #show_img(cat, 'cat')
    #cap = cv.VideoCapture('data/tree.avi')
    #while 1:
    #    ret, frame = cap.read()
    #    if not ret:
    #        break
    #    cv.imshow('frame', frame)
    #    if cv.waitKey(40) == 27:
    #        break
    #cap.release()

    top, bottom, left, right = (50, 50, 50, 50)
    # 复制图像最边缘的像素
    replicate = cv.copyMakeBorder(rgb_img, top, bottom,
                                  left, right, borderType=cv.BORDER_REPLICATE)
    # 反射，对感兴趣的图像的像素在两边进行复制
    reflect = cv.copyMakeBorder(rgb_img, top, bottom,
                                left, right, borderType=cv.BORDER_REFLECT)
    # 反射法
    reflect101 = cv.copyMakeBorder(rgb_img, top, bottom,
                                   left, right, borderType=cv.BORDER_REFLECT_101)
    # 外包装法
    wrap = cv.copyMakeBorder(rgb_img, top, bottom,
                             left, right, borderType=cv.BORDER_WRAP)
    # 常数填充
    constant = cv.copyMakeBorder(rgb_img, top,
                                 bottom, left, right,
                                 borderType=cv.BORDER_CONSTANT, value=[0, 0, 255])
    cv.imshow('constant', constant)
    # PIL的形状是[h, w, rgb], 不是opencv的[h, w, bgr]
    plt.subplot(231)
    plt.imshow(rgb_img, 'gray')
    plt.title('ORIGINAL')
    plt.subplot(232)
    plt.imshow(replicate, 'gray')
    plt.title('replicate')
    plt.subplot(233)
    plt.imshow(reflect, 'gray')
    plt.title('reflect')
    plt.subplot(234)
    plt.imshow(reflect101, 'gray')
    plt.title('reflect101')
    plt.subplot(235)
    plt.imshow(wrap, 'gray')
    plt.title('wrap')
    plt.subplot(236)
    plt.imshow(constant, 'gray')
    plt.title('constant')
    plt.show()

    football = cv.imread('data/messi5.jpg')
    ball = football[280:340,330:390,:]
    football[273:333, 100:160] = ball
    cv.imshow('double', football)

    # 图像融合，使用addWeighted函数，必须是相同shape的才可以进行操作
    img1 = cv.imread('data/WindowsLogo.jpg')
    img2 = cv.imread('data/LinuxLogo.jpg')
    dest = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
    cv.imshow('Blending', dest)

    cv.waitKey(0)
    cv.destroyAllWindows()