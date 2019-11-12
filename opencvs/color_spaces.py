# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


def color_space(imgobj):
    # 常见的色彩空间包括RGB，HSV，HIS，YCrCb(皮肤检测)和YUV
    # 其中，H是1-180，S 0-255，V 0-255
    gray = cv.cvtColor(imgobj, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    hsv = cv.cvtColor(imgobj, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', imgobj)


def extract_obj():
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print('Cannot open the camera')
        return
    while 1:
        ret, frame = capture.read()
        if not ret:
            print('Cannot receive, exit....')
            break
        cv.imshow('video', frame)

        # 颜色提取，提取绿色的信息
        # hsv色彩空间比较容易提取色彩信息，可以有很大的差异化信息
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 43, 46])
        high_hsv = np.array([77, 255, 255])
        mask = cv.inRange(hsv, lower_hsv, upperb=high_hsv)
        cv.imshow('mask', mask)

        # rgb通道拆分
        b, g, r = cv.split(frame)
        cv.imshow('blue', b)
        cv.imshow('green', g)
        cv.imshow('red', r)

        # 合并通道
        copy = cv.merge([b,g,r])
        cv.imshow('copy', copy)

        # 减去B通道的信息
        frame[:,:,0] = 0
        cv.imshow('no blue', frame)

        if 27 == cv.waitKey(1):
            break
    capture.release()

src = cv.imread('1.jpg')
color_space(src)
cv.waitKey(0)
extract_obj()
cv.destroyAllWindows()