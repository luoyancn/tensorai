import cv2 as cv


cap = cv.VideoCapture(0)
cap.open('data/tree.avi')
while 1:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow('frame', frame)
    cv.waitKey(400)

cap.release()
cv.destroyAllWindows()