# -*- coding: utf-8 -*-

# 泛洪填充实际上就是颜色的填充
# ROI就是图像中感兴趣的区域，即region of interesting


import cv2 as cv
import numpy as np


def roi_opt(srcimg):
    # 读取图像的x，从200到400像素，y的200像素到400像素
    face = srcimg[200:400, 200:400]
    # 转换为灰度图像
    gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

    # 将灰度图像还原为RGB 3通道图像，但是还原后的图像，不一定就是彩色图像!!
    dest = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # 对原始图像进行覆盖操作，[200:400, 200:400]即是感兴趣的区域
    srcimg[200:400, 200:400] = dest


def fill_color(srcimg):
    _copy = srcimg.copy()
    height, width = srcimg.shape[:2]
    # mask做填充时，大小必须是+2，并且类型必行是uint8，否则会出现问题
    mask = np.zeros([height + 2, width + 2], np.uint8)
    # imgobj和_copy是同一个对象
    # floodFill第3个参数为seedpoint，表示从该像素点出发，查找周围的所有像素点
    # 第4个参数表示颜色
    # 第5个参数为loDiff，表示低值，表示当前的观察点像素值与其相邻区域像素值或待加入该区域的像素之间的亮度或颜色之间负差的最大值
    # 第6个参数为upDiff，表示填充的高值, 表示当前的观察点像素值与其相邻区域像素值或待加入该区域的像素之间的亮度或颜色之间负差的最小值
    # 填充的规则如下：
    # src(seed.x’, seed.y’) - loDiff <= src(x, y) <= src(seed.x’, seed.y’) +upDiff
    # 第7个参数为flag，FLOODFILL_FIXED_RANGE表示在范围内的全局进行填充,RGB图像必须
    # FLOODFILL_MASK_ONLY表示针对mask进行填充
    ret, imgobj, _mask, _rect = cv.floodFill(
        _copy, mask, (30, 30), (0, 255, 255), (100, 100, 100),
        (50, 50, 50), flags=cv.FLOODFILL_FIXED_RANGE)
    return imgobj


def fill_binary():
    img = np.zeros([400, 400, 3], np.uint8)
    img[100:300, 100:300, :] =  255
    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0 # 所有为0的区域会被填充
    # 奇怪的问题：下面3行代码的结果一样
    #ret, imgobj, _mask, _rect = cv.floodFill(
    #    img, mask, (200, 200), (0, 0, 255), loDiff=cv.FLOODFILL_MASK_ONLY)
    #ret, imgobj, _mask, _rect = cv.floodFill(
    #    img, mask, (200, 200), (0, 0, 255), upDiff=cv.FLOODFILL_MAKS_ONLY)
    ret, imgobj, _mask, _rect = cv.floodFill(
        img, mask, (200, 200), (0, 0, 255), flags=cv.FLOODFILL_FIXED_RANGE)
    return imgobj


if __name__ == '__main__':
    src = cv.imread('data/lena.jpg')
    img = src.copy()
    roi_opt(img)
    #cv.imshow('dest', img)
    cv.imshow('fill_color', fill_color(img))
    cv.imshow('file_binary', fill_binary())

    cv.waitKey(0)
    cv.destroyAllWindows()