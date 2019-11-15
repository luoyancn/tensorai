# -*- coding: utf-8 -*-

import cv2 as cv


cap = cv.VideoCapture(0)

# 设置镜头分辨率
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
# 20表示视频的帧率
output = cv.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

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