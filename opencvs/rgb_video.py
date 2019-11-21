# -*- coding: utf-8 -*-

import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera')
    exit()

while 1:
    ret , frame = cap.read()
    if not ret:
        print('Cannot receive fram from camera, exiting...')
        break
    frame = cv.flip(frame, 1)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary_frame = cv.threshold(gray_frame, 0, 255,
                                     cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, trunc_binary = cv.threshold(gray_frame, 128, 255, cv.THRESH_TRUNC)
    mean_binary = cv.adaptiveThreshold(gray_frame, 255,
        cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow('rgb_frame', frame)
    cv.imshow('binary_frame', binary_frame)
    cv.imshow('trunc_frame', trunc_binary)
    cv.imshow('mean_frame', mean_binary)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()