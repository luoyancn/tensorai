# -*- coding: utf-8 -*-

import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1280)
if not cap.isOpened():
    print('Cannot open camera')
    exit()

while 1:
    ret , frame = cap.read()
    if not ret:
        print('Cannot receive fram from camera, exiting...')
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()