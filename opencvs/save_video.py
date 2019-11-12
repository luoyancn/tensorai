# -*- coding: utf-8 -*-

import cv2 as cv


cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
output = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Cannot receive from camerai, exit....')
        break
    # 画面翻转
    #frame = cv.flip(frame, 0)
    # 镜像模式
    frame = cv.flip(frame, 1)
    output.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
output.release()
cv.destroyAllWindows()