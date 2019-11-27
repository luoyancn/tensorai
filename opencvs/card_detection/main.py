# -*- coding: utf-8 -*-

# 信用卡数字识别
# 主要使用的技术包括：
# 1.模板匹配
# 2.形态学处理，膨胀与腐蚀
# 3.轮廓检测，外接形状

import argparse

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True,
                        help='The path of input card image')
    parser.add_argument('-t', '--template', required=True,
                        help='The path of template OCR-A image')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img = args.image
    tpl = args.template
    debug = args.debug

    img = utils.read_img(img)
    tpl = utils.read_img(tpl)

    tpl_binary = utils.binary_img(
        tpl, min_val=10, max_val=255,
        thresh_type=utils.THRESH_TYPE['binary_inv'])[1]

    ref_cnts, _ = utils.img_contours(tpl, tpl_binary)

    digits = utils.roi_boundings(ref_cnts, tpl_binary)

    utils.processing(img, digits)