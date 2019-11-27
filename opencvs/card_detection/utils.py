# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


FIRST_NUMBER = {
    '3': 'American Express',
    '4': 'Visa',
    '5': 'MasterCard',
    '6': 'Discover Card'
}


THRESH_TYPE = {
    'binary': cv.THRESH_BINARY,
    'binary_inv': cv.THRESH_BINARY_INV,
    'tozero': cv.THRESH_TOZERO,
    'tozero_inv': cv.THRESH_TOZERO_INV,
    'trunc': cv.THRESH_TRUNC
}


rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
sq_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))


def showimg(img, imgname, debug=False):
    cv.imshow(imgname, img)
    timeout = 1000
    if debug:
        timeout = 0
    cv.waitKey(timeout)
    cv.destroyWindow(imgname)


def read_img(imgpath, gray=False):
    return cv.imread(
        imgpath, flags=cv.IMREAD_GRAYSCALE if gray else cv.IMREAD_UNCHANGED)


def binary_img(img, min_val=0, max_val=255, thresh_type=cv.THRESH_BINARY):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.threshold(gray, min_val, max_val, thresh_type)


def img_contours(origin_img, binary_img):
    img_cnts, _ = cv.findContours(
        binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(origin_img, img_cnts, -1, (0, 0, 255), 3)
    return _sort_contours(img_cnts)


def _sort_contours(cnts):
    boundingboxs = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingboxs) = zip(*sorted(zip(
        cnts, boundingboxs), key=lambda b: b[1][0]))
    return (cnts, boundingboxs)


def roi_boundings(contours, binary_img):
    digits = {}
    for idx, cnt in enumerate(contours):
        x_start, y_start, wight, height = cv.boundingRect(cnt)
        roi = binary_img[y_start:y_start+height, x_start:x_start+wight]
        roi = cv.resize(roi, (57, 88))
        digits[idx] = roi
    return digits


def processing(src_rgb_img, digits):
    img = resize(src_rgb_img, width=300)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    top_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rect_kernel)
    gradx = cv.Sobel(top_hat, cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradx = np.absolute(gradx)
    min_val, max_val = np.min(gradx), np.max(gradx)
    gradx = (255*((gradx-min_val)/(max_val-min_val)))
    gradx = gradx.astype('uint8')

    gradx = cv.morphologyEx(gradx, cv.MORPH_CLOSE, rect_kernel)
    thresh = cv.threshold(gradx, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]

    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sq_kernel)

    thresh_cnts, _ = cv.findContours(
        thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cur_img = img.copy()
    cv.drawContours(cur_img, thresh_cnts, -1, (0, 0, 255), 3)

    locs = []
    for idx, _cnt in enumerate(thresh_cnts):
        x, y, w, h = cv.boundingRect(_cnt)
        ar = w / float(h)
        if 2.5 < ar < 4.0:
            if 40 < w < 55 and 10 < h < 20:
                locs.append((x, y, w, h))
    locs = sorted(locs, key=lambda x:x[0])

    output = []
    for idx, (gx, gy, gw, gh) in enumerate(locs):
        group_output = []
        group = gray[gy-5:gy+gh+5, gx-5:gx+gw+5]
        group = cv.threshold(group, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
        digits_cnts, _ = cv.findContours(
            group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        digits_cnts = _sort_contours(digits_cnts)[0]
        for c in digits_cnts:
            x, y, w, h = cv.boundingRect(c)
            _roi = group[y:y+h, x:x+w]
            _roi = cv.resize(_roi, (57, 88))

            scores = []
            for digit, digitsROI in digits.items():
                result = cv.matchTemplate(_roi, digitsROI, cv.TM_CCOEFF)
                _, score,_, _ = cv.minMaxLoc(result)
                scores.append(score)
            group_output.append(str(np.argmax(scores)))
        cv.rectangle(img, (gx-5, gy-5), (gx+gw+5, gy+gh+5), (0,0, 255), 1)
        cv.putText(img, ''.join(group_output), (gx, gy-5),
                   cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.65, (0,0,255), 2)
        output.extend(group_output)
    print('The card number is : {}'.format(''.join(output)))
    showimg(img, 'img', debug=True)


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized