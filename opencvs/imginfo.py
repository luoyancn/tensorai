# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


def get_image_info(imgobj):
    return imgobj.shape, imgobj.size, imgobj.dtype


def get_video_info(debug=False):
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print('Cannot open the camera')
        try:
            capture.release()
        except Exception as ex:
            pass
        return
    while 1:
        ret, frame = capture.read()
        if not ret:
            print('Cannot receive from camera, exit....')
            break
        frame = cv.flip(frame, 1)
        cv.imshow('video', frame)
        shape, size, dtype = get_image_info(frame)
        if debug:
            print('The video shape is %r, size is %d, and dtype is %r' %(
                shape, size, dtype))
        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


def access_pixels(imgobj):
    height = imgobj.shape[0]
    width = imgobj.shape[1]
    channels = imgobj.shape[2]
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = imgobj[row, col, c]
                imgobj[row, col, c] = 255 - pv
    cv.imshow('access_pixels', imgobj)


def create_image():
    img = np.zeros([400, 400, 3], np.uint8)
    # 第3个维度表示RGB通道， 0为B, 1为G，2为R
    img[:, :, 2] = np.ones([400, 400]) * 255
    cv.imshow('fromnumpy', img)


def create_gray_image():
    img = np.zeros([400, 400], np.uint8)
    #img = img * 127
    img[:, :] = np.ones([400, 400]) * 127
    cv.imshow('fromnumpy_gray', img)


def inverse(imgobj):
    dst = cv.bitwise_not(imgobj)
    cv.imshow('inverse', dst)


src = cv.imread('1.jpg')
#cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
#cv.imshow('input', src)

shape, size, dtype = get_image_info(src)
print('The image info, shape %r, size %d, dtype %r' %(shape, size, dtype))

inverse(src)
access_pixels(src)
create_image()
create_gray_image()
cv.waitKey(0)
cv.destroyAllWindows()

#get_video_info()