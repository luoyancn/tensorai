# -*- coding: utf-8 -*-

import cv2 as cv
from matplotlib import pyplot as plt


src_img = cv.imread('1.jpg')
gracy_img = cv.imread('1.jpg', 0)
#cv.namedWindow('src_image', cv.WINDOW_AUTOSIZE)
print(src_img)
cv.imshow('input image', src_img)
cv.imshow('gracy image', gracy_img)
cv.waitKey(0)

cv.namedWindow('src_image', cv.WINDOW_AUTOSIZE)
cv.imshow('src_image', src_img)
cv.imwrite('output.png', gracy_img)

# 使用matplotlib进行图片的显示
plt.imshow(gracy_img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

cv.destroyAllWindows()