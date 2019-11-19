# -*- coding: utf-8 -*-

# 验证码识别

import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess


def text_detected(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    _kerel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
    open_out = cv.morphologyEx(bin1, cv.MORPH_OPEN, _kerel)

    cv.bitwise_not(open_out, open_out)
    text_img = Image.fromarray(open_out)
    text = tess.image_to_string(text_img)
    print('result is %s' % text)


if __name__ == '__main__':
    src = cv.imread('data/0.jpg')
    text_detected(src)