# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print('Cannot open the camera')
    exit(0)

while 1:
    ret, frame = cam.read()
    if not ret:
        print('Cannot receive from camera')
        break
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    xdest = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    xdest = cv.convertScaleAbs(xdest)
    ydest = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    ydest = cv.convertScaleAbs(ydest)
    good_xydest = cv.addWeighted(xdest, 0.5, ydest, 0.5, 0)
    cv.imshow('video', good_xydest)
    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()