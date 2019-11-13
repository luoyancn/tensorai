# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

# 像素可以进行四则运算，进行亮度和对比度的调节
# 逻辑运算则是应用于遮罩层控制


# 图像增加
def img_add_opt(img1, img2):
    dest = cv.add(img1, img2)
    return dest


# 图像减法
def img_sub_opt(img1, img2):
    dest = cv.subtract(img1, img2)
    return dest


def img_divide_opt(img1, img2):
    return cv.divide(img1, img2)


def img_mul_opt(img1, img2):
    return cv.multiply(img1, img2)


def img_mean_opt(img):
    # 图像的均值和方差
    # 方差越大，说明图像的差异性越大，换言之就是对比度越明显
    # 如果均值和方差结果不大，说明没有加载有效信息，或者图像本身没有有效信息
    _mean, _std = cv.meanStdDev(img)
    return cv.mean(img), _mean, _std


# 图像的逻辑预算
def logic_opt(img1, img2):
    return cv.bitwise_and(img1, img2), cv.bitwise_not(img1, img2),\
         cv.bitwise_or(img1, img2), cv.bitwise_xor(img1, img2)


# 调整图像的亮度, C表示对比度，b表示亮度
# alpha通道对应的是对比度？？？
def contrast_brightness(img, c, b):
    height, width, channel = img.shape
    blank = np.zeros([height, width, channel], img.dtype)
    return cv.addWeighted(img, c, blank, 1-c, b)

if __name__ == '__main__':
    wlogo = cv.imread('data/WindowsLogo.jpg')
    llogo = cv.imread('data/LinuxLogo.jpg')

    cv.imshow('add opt', img_add_opt(wlogo, llogo))
    cv.imshow('sub opt', img_sub_opt(llogo, wlogo))
    cv.imshow('div opt', img_divide_opt(llogo, wlogo))
    cv.imshow('mul opt', img_mul_opt(wlogo, llogo))

    img_mean, _mean, _std = img_mean_opt(wlogo)
    cv.imshow('mean opt', img_mean)
    print('%r,\n %r'%(_mean, _std))

    cv.imshow('bright', contrast_brightness(wlogo, 0.8, 0))

    #and_, not_, or_, xor_ = logic_opt(llogo, wlogo)
    #cv.imshow('and opt', and_)
    #cv.imshow('not opt', not_)
    #cv.imshow('or opt', or_)
    #cv.imshow('xor opt', xor_)


    cv.waitKey(0)
    cv.destroyAllWindows()