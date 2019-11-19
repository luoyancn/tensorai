# -*- coding: utf-8 -*-

# 人脸检测


import cv2 as cv
import numpy as np


def face_detected(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(
        'data/haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


if __name__ == '__main__':
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
        face_detected(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()