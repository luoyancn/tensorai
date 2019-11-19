# -*- coding: utf-8 -*-

# 物体测量， 弧长与面积
# 计算每个轮廓的弧长与面积，像素单位
# 获取轮廓的多边形拟合结果
# approxPolyDP函数，epsilon越小，折线越逼近真实形状， close表示是否为闭合区域

import cv2 as cv
import numpy as np


def mean_object(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    dest = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    contours, hireachy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        # 外接矩形
        x, y, w, h = cv.boundingRect(contour)
        # 宽高比
        rate = min(w, h) / max(w, h)
        mm = cv.moments(contour)
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv.circle(img, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print('contour area %f and rate is %f' % (area, rate))
        appres = cv.approxPolyDP(contour, 2, True)
        print(appres.shape)
        if appres.shape[0] > 0:
            cv.drawContours(dest, contours, 1, (0, 255, 0), 2)
    cv.imshow('dest', dest)


if __name__ == '__main__':
    src = cv.imread('data/detect_blob.png')

    mean_object(src)
    cv.imshow('mean_object', src)

    cv.waitKey(0)
    cv.destroyAllWindows()