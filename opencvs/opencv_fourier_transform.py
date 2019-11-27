# -*- coding: utf-8 -*-

# 傅立叶变换
# 高频：变化剧烈的灰度分量，比如边界
# 低频：变化缓慢的灰度分量，比如大海
# 低通滤波器：保留低频，使得图像模糊
# 高通滤波器：保留高频，使得图像细节增强
# opencv当中，需要先将图像转换为np.float32格式，常用函数为dft和idft
# 得到的结果当中频率为0的部分会在左上角，通常需要转换到中心位置，可以使用shift变换实现
# dft返回的结果是双通道的（实部和虚部），需要转换成图像格式才能显示(0, 255)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


rgb_lena = cv.imread('data/lena.jpg')
gray_lena = cv.cvtColor(rgb_lena, cv.COLOR_BGR2GRAY)
float32_lena = np.float32(gray_lena)
# 执行傅立叶变换
dft_img = cv.dft(float32_lena, flags=cv.DFT_COMPLEX_OUTPUT)
# 进行shift操作，得到频谱图
dft_shift = np.fft.fftshift(dft_img)
# 将频谱图转换为能够识别的灰度图像格式，转换为0-255之间
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:,:, 0], dft_shift[:, :, 1]))

plt.subplot(121)
plt.imshow(gray_lena, cmap='gray')
plt.title('input')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude')
plt.xticks([])
plt.yticks([])
plt.show()

rows, cols = gray_lena.shape
# 获取图像的中心位置
center_row, center_col = int(rows/2), int(cols/2)

# 低通滤波，双通道
mask = np.zeros((rows, cols, 2), np.uint8)
mask[center_row-30:center_row+30, center_col-30:center_col+30] =  1

# idft
fshift = dft_shift * mask
# shift变换，还原原始位置
f_ishift = np.fft.ifftshift(fshift)
# idft进行图像还原
img_back = cv.idft(f_ishift)
# 将频域的双通道转换为灰度图像
img_back = cv.magnitude(img_back[:, :, 0], img_back[:,:, 1])
plt.subplot(121)
plt.imshow(gray_lena, cmap='gray')
plt.title('input')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(img_back, cmap='gray')
plt.title('result')
plt.xticks([])
plt.yticks([])
plt.show()

# 高通滤波
_mask = np.ones((rows, cols, 2), np.uint8)
_mask[center_row-30:center_row+0, center_col-30:center_col+30] = 0
_fshift = dft_shift * _mask
_f_ishift = np.fft.ifftshift(_fshift)
_img_bak = cv.idft(_f_ishift)
_img_bak = cv.magnitude(_img_bak[:,:,0], _img_bak[:,:, 1])
plt.subplot(121)
plt.imshow(gray_lena, cmap='gray')
plt.title('input')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(_img_bak, cmap='gray')
plt.title('result')
plt.xticks([])
plt.yticks([])
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()