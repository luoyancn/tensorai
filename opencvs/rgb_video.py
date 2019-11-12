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
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()