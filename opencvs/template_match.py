# -*- coding: utf-8 -*-

# 模板匹配，通常在工业领域比较常用。实际比较复杂的场景当中比较难
# 在图像当中查找与模板相似的区域，效率并不是特别高

import cv2 as cv
import numpy as np


def template():
    tpl = cv.imread('data/templ.png')
    target = cv.imread('data/pic1.png')
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    tpl_height, tpl_width = tpl.shape[:2]
    for method in methods:
        result = cv.matchTemplate(target, tpl, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if cv.TM_SQDIFF:
            tpl_l = min_loc
        else:
            tpl_l = max_loc
        br = (tpl_l[0] + tpl_width, tpl_l[1] + tpl_height)
        cv.rectangle(target, tpl_l, br, (0,0,255), 2)
        cv.imshow('match-'+str(method), target)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    template()